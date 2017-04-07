import torch as t
import torch.nn as nn
import numpy as np
from torch_modules.other.embedding_lockup import EmbeddingLockup
from torch.autograd import Variable
from model.sequence_to_image import SequenceToImage
from model.image_to_sequence import ImageToSequence
from model.wasserstein_discriminator import WassersteinDiscriminator
import torch.nn.functional as F


class CDVAE(nn.Module):
    def __init__(self, params, path_prefix):
        super(CDVAE, self).__init__()

        self.params = params
        self.path_prefix = path_prefix

        self.embeddings = EmbeddingLockup(self.params, self.path_prefix)

        """
        takes batch size of sequences and sample appropriate images
        discriminator network uses to make images more realistic
        """
        self.seq2image = SequenceToImage(params)
        self.discr = WassersteinDiscriminator(params, self.path_prefix)

        """
        takes array of images of batch size length to emit batch size of sequences
        model uses decoder context input in pair with latent representation
        """
        self.image2seq = ImageToSequence(params, self.path_prefix)

    def forward(self, drop_prob=0,
                encoder_word_input=None, encoder_character_input=None,
                target_images=None, target_image_sizes=None,
                decoder_word_input=None):
        """
        :param drop_prob: probability of an element of decoder input to be zeroed in sense of dropout 
        :param encoder_word_input: An tensor with shape of [batch_size, seq_len] of Long type
        :param encoder_character_input: An tensor with shape of [batch_size, seq_len, max_word_len] of Long type
        :param target_images: array of batch_size length of images paths
        :param target_image_sizes: sizes of target images 
        :param decoder_word_input: An tensor with shape of [batch_size, max_seq_len + 1] of Long type
        
        This method is necessary to forward propagate of both seq_to_image and image_to_seq models
        In order to sample data from decoders of these models use :sample_images: and :sample_seq: methods
        """

        seq_to_image_result = self.seq2image(self.embeddings,
                                             drop_prob=drop_prob,
                                             encoder_word_input=encoder_word_input,
                                             encoder_character_input=encoder_character_input,
                                             target_sizes=target_image_sizes)

        image_to_seq_result = self.image2seq(self.embeddings,
                                             drop_prob=drop_prob,
                                             encoder_image_input=target_images,
                                             decoder_input=decoder_word_input)

        return seq_to_image_result, image_to_seq_result

    def seq2image_parameters(self):
        return [p for p in self.seq2image.parameters() if p.requires_grad]

    def image2seq_parameters(self):
        return [p for p in self.image2seq.parameters() if p.requires_grad]

    def discr_parameters(self):
        return [p for p in self.discr.parameters() if p.requires_grad]

    def trainer(self, s2i_optimizer, i2s_optimizer, disc_optimizer, batch_loader):
        def train(batch_size, num_discr_updates, use_cuda, drop_prob):
            """
            :param batch_size: batch size 
            :param use_cuda: whether to use cuda
            :param drop_prob: drop probability
            
            propagate seq2image, image2seq and discriminator networks and updates them in appropriate way
            """

            word_encoder_input, character_encoder_input, images_input, \
                images_input_sizes, word_decoder_input, word_decoder_target = \
                batch_loader.next_batch(batch_size, 'train')

            # update discriminator network
            for i in range(num_discr_updates):
                z = t.randn([batch_size, self.params.latent_variable_size])

                image_out = self.sample_images(z, images_input_sizes, use_cuda)
                d_loss, _ = self.discr(image_out, true_data=batch_loader.sample_real_examples(batch_size))

                disc_optimizer.zero_grad()
                d_loss.backward()
                disc_optimizer.step()

                for p in self.discr.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # update both sequence to image and image to sequence models
            (out_s2i, kld_s2i, (mu_s2i, logvar_s2i)), (out_i2s, _, kld_i2s, (mu_i2s, logvar_i2s)) \
                = self(drop_prob, word_encoder_input, character_encoder_input,
                       images_input, images_input_sizes, word_decoder_input)

            reconst_loss_s2i = SequenceToImage.mse(out_s2i, images_input)
            reconst_loss_i2s = self.image2seq.cross_entropy(out_i2s, word_decoder_target)
            kld_id_loss = t.pow(mu_i2s - mu_s2i, 2).mean() + t.pow(logvar_i2s - logvar_s2i, 2).mean()
            _, g_loss_s2i = self.discr(out_s2i, true_data=batch_loader.sample_real_examples(batch_size))

            """
            both losses are constructed from reconstruction loss, KL-distance loss
            and kld-identity loss that forces models to have the same q(z|x) and q(z|y) 
            where x and y are domains of learning
            
            sequence to image model also provided with generation loss from WGAN model
            to make sampled data look better
            """
            loss_s2i = reconst_loss_s2i + kld_s2i + kld_id_loss + g_loss_s2i
            loss_i2s = reconst_loss_i2s + kld_i2s + kld_id_loss

            s2i_optimizer.zero_grad()
            loss_s2i.backward(retain_variables=True)
            s2i_optimizer.step()

            i2s_optimizer.zero_grad()
            loss_i2s.backward()
            i2s_optimizer.step()

            return (reconst_loss_s2i, kld_s2i, g_loss_s2i), (reconst_loss_i2s, kld_i2s), kld_id_loss

        return train

    def sample_images(self, z, target_sizes, use_cuda, to_numpy=False):
        z = Variable(z)
        if use_cuda:
            z = z.cuda()

        result = self.seq2image(embeddings=self.embeddings, target_sizes=target_sizes, z=z)[0]

        return [var.data.cpu().numpy() for var in result] if to_numpy else result

    def sample_seq(self, batch_loader, seq_len, z, use_cuda):
        z = Variable(z)

        decoder_input = batch_loader.go_input(1)
        decoder_input = Variable(t.from_numpy(decoder_input))
        if use_cuda:
            z = z.cuda()
            decoder_input = decoder_input.cuda()

        result = ''
        initial_state = None

        for i in range(seq_len):
            logits, final_state, _, _ = self.image2seq(embeddings=self.embeddings,
                                                       decoder_input=decoder_input, initial_state=initial_state, z=z)

            logits = logits.view(-1, self.params.word_vocab_size)
            prediction = F.softmax(logits)

            word = batch_loader.sample_word_from_distribution(prediction.data.cpu().numpy()[-1])

            if word == batch_loader.stop_token:
                break

            result += ' ' + word

            decoder_input = np.array([[batch_loader.word_to_idx[word]]])
            decoder_input = Variable(t.from_numpy(decoder_input).long())

            if use_cuda:
                decoder_input = decoder_input.cuda()

        return result

    def sample(self, batch_loader, target_sizes, seq_len, z, use_cuda):
        z = t.from_numpy(z).float()
        return self.sample_images(z, target_sizes, use_cuda, True), self.sample_seq(batch_loader, seq_len, z, use_cuda)
