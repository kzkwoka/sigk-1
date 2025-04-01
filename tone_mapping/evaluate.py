import cv2
from tqdm import tqdm
from model import TMNet
from dataset import TMDataset
from utils import read_exr, tone_map_reinhard, tone_map_mantiuk, evaluate_image



if __name__ == '__main__':
    # path = "logs/SIGK-2/crjclu06/checkpoints/epoch=499-step=6000.ckpt"
    path = "logs/SIGK-2/3mbpnd2p/checkpoints/epoch=499-step=90500.ckpt"
    model = TMNet.load_from_checkpoint(path, in_channels=3)
    model.eval()

    dataset = TMDataset('tone_mapping/sihdr/resized')

    b_net, b_reinhard, b_mantiuk = 0, 0, 0
    for i_low, i_mid, i_high, i_mu, i_src in tqdm(dataset):

        image = i_src

        mapped = model(i_low.unsqueeze(0).cuda(), i_mid.unsqueeze(0).cuda(), i_high.unsqueeze(0).cuda())
        mapped = mapped.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

        tone_mapped_reinhard = tone_map_reinhard(image)
        tone_mapped_mantiuk = tone_map_mantiuk(image)

        b_net += evaluate_image(image=mapped)
        b_reinhard += evaluate_image(image=tone_mapped_reinhard)
        b_mantiuk += evaluate_image(image=tone_mapped_mantiuk)
    print('BRISQUE net', b_net / len(dataset))
    print('BRISQUE reinhard', b_reinhard / len(dataset))
    print('BRISQUE mantiuk', b_mantiuk / len(dataset))

    # cv2.imshow('original', image)
    # cv2.imshow('tone_mapped_net', mapped)
    # cv2.imshow('tone_mapped_reinhard', tone_mapped_reinhard)
    # cv2.imshow('tone_mapped_mantiuk', tone_mapped_mantiuk)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()