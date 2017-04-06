import torch as t
from torch.autograd import Variable
from scipy import misc
from model.sequence_to_image import SequenceToImage
from model.image_to_sequence import ImageToSequence
from utils.batchloader import BatchLoader
from utils.parameters import Parameters
from torch_modules.other.embedding_lockup import EmbeddingLockup
from model.encoders.image_encoder import ImageEncoder
from model.encoders.text_encoder import TextEncoder
from model.wasserstein_discriminator import WassersteinDiscriminator
from torch_modules.other.expand_with_zeros import expand_with_zeroes
from model.cdvae import CDVAE
from utils.functions import *

if __name__ == '__main__':
    batch_loader = BatchLoader('../')
    parameters = Parameters(batch_loader.max_word_len,
                            batch_loader.max_seq_len,
                            batch_loader.words_vocab_size,
                            batch_loader.chars_vocab_size)

    batch_size = 5

    print()

    embedding = EmbeddingLockup(parameters, '../')
    print('embeddings is initialized 👍\n')

    word_level_encoder_input, character_level_encoder_input, target_images, \
    real_images, target_images_sizes, decoder_text_input, decoder_text_target = \
        batch_loader.next_batch(batch_size, 'train')

    discriminator = WassersteinDiscriminator(parameters, '../')
    print('discriminator is initialized 👍\n')
    gen = [Variable(t.rand([3, 500, 500])) for _ in range(batch_size - 3)] + \
          [Variable(t.rand([3, 400, 450])) for _ in range(3)]
    d_loss, g_loss = discriminator(gen, real_images)
    assert d_loss.size()[0] == g_loss.size()[0] == 1 and len(d_loss.size()) == len(g_loss.size()) == 1, \
        'invalid discriminator output size ⛔'
    print('discriminator is valid 👍\n')
    del discriminator

    image_encoder = ImageEncoder(parameters, '../')
    print('image encoder is initialized 👍\n')
    image_encoder_result = image_encoder(target_images)
    assert image_encoder_result[0] == len(target_images) and \
           image_encoder_result[1] == 512, 'invalid image encoder size ⛔'
    print('image encoder is valid 👍\n')
    del image_encoder

    sequence_to_image = SequenceToImage(parameters)
    print('sequence to image is initialized 👍\n')
    z = Variable(t.rand([batch_size, parameters.latent_variable_size]))
    out, kld, (mu, logvar) = sequence_to_image(embedding, target_sizes=target_images_sizes, z=z)

    assert all([var is None for var in [kld, mu, logvar]])
    assert all([predicat
                for i, var in enumerate(out) for predicat in list(var.size()[1:] == target_images_sizes[i])]), \
        'invalid out size of images ⛔'

    out, kld, (mu, logvar) = sequence_to_image(embedding,
                                               encoder_word_input=word_level_encoder_input,
                                               encoder_character_input=character_level_encoder_input,
                                               target_sizes=target_images_sizes,
                                               z=None)
    assert len(out) == mu.size()[0] == logvar.size()[0] == batch_size, 'invalid out size ⛔'
    assert all([predicat
                for i, var in enumerate(out) for predicat in list(var.size()[1:] == target_images_sizes[i])]), \
        'invalid out size of images ⛔'
    print('sequence to image tests have passed 👍\n')
    del sequence_to_image

    image_to_sequence = ImageToSequence(parameters, '../')
    print('image to sequence is initialized 👍\n')
    out, final_state, kld, (mu_2, logvar_2) = image_to_sequence(embedding, 0, target_images, decoder_text_input)
    assert all([len(size) == 2 for size in [mu_2.size(), mu.size(), logvar_2.size(), logvar.size()]]), \
        'invalid mu and logvar rang ⛔\n'
    out, final_state, kld, (mu_2, logvar_2) = image_to_sequence(embedding, decoder_input=decoder_text_input, z=z)
    assert len(out.size()) == 3, 'invalid out size ⛔\n'
    [b, sl, ws] = out.size()
    assert b == batch_size and sl == decoder_text_input.size()[1] and ws == parameters.word_vocab_size, \
        'ivalid decoder out size ⛔\n'
    print('image to sequence tests have passed 👍\n')

    cdvae = CDVAE(parameters, '../')
    seq_to_image_result, image_to_seq_result = cdvae(0, word_level_encoder_input, character_level_encoder_input,
                                                     target_images, target_images_sizes, decoder_text_input)
    assert len(seq_to_image_result) == 3 and len(image_to_seq_result) == 4
    assert len(seq_to_image_result[2]) == 2 and len(image_to_seq_result[3]) == 2
    print('CDVAE tests passed 👍\n')
