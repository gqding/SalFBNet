import argparse
import os
from dataset import get_loader, get_loader_test
from solver import Solver


def main(config):
    if config.mode == 'train':
        pass
    elif config.mode == 'test':
        test_loader = get_loader_test(config.test_path, config.test_label, config.img_size, config.batch_size,
                                      mode='test',
                                      filename=config.test_file, num_thread=config.num_thread)
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, None, test_loader, config)
        test.test()

    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--clip_gradient', type=float, default=1.0)
    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--multi_gpu', type=bool, default=True)
    parser.add_argument('--vgg', type=str, default='')
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--label_path', type=str, default='')
    parser.add_argument('--img_size', type=int, default=None)  # 256
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)  # 8
    parser.add_argument('--val', type=bool, default=True)
    parser.add_argument('--val_path', type=str, default='')
    parser.add_argument('--val_label', type=str, default='')

    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./results')
    parser.add_argument('--epoch_val', type=int, default=1)
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--backbone', type=str, default='Res18')  # Res18, Res18Fixed
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--test_label', type=str, default='')
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--test_fold', type=str, default='')
    parser.add_argument('--use_crf', type=bool, default=False)

    # Misc
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)

    config = parser.parse_args()
    if config.test_file is None:
        if 'SALICON/images/test' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Test_SALICON')
        elif 'SALICON/images/val' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Val_SALICON')
        elif 'MIT1003/val' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Val_MIT1003')
        elif 'MIT1003/all' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'All_MIT1003')
        elif 'MIT300' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Test_MIT300')
        elif 'CAT2000' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Test_CAT2000')
        elif 'PseudoSal' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Val_PseudoSal')
        elif 'DUT-OMRON' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Test_DUT-OMRON')
        elif 'PASCAL-S' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Test_PASCAL-S')
        elif 'TORONTO' in config.test_path:
            config.test_fold = os.path.join(config.save_fold, 'Test_TORONTO')
        else:
            raise NotImplementedError
    else:
        if 'Eye_Fixation_Test_SALICON' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Test_SALICON')
        elif 'Eye_Fixation_Val_SALICON' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Val_SALICON')
        elif 'Eye_Fixation_Val_MIT1003' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Val_MIT1003')
        elif 'Eye_Fixation_All_MIT1003' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'All_MIT1003')
        elif 'Eye_Fixation_Test_MIT300' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Test_MIT300')
        elif 'Eye_Fixation_Test_CAT2000' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Test_CAT2000')
        elif 'Val_Eye_Fixation_PseudoSal_all' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Val_PseudoSal')
        elif 'Eye_Fixation_Train_DUT-OMRON' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Test_DUT-OMRON')
        elif 'Eye_Fixation_Train_PASCAL-S' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Test_PASCAL-S')
        elif 'Eye_Fixation_Train_TORONTO' in config.test_file:
            config.test_fold = os.path.join(config.save_fold, 'Test_TORONTO')
        else:
            raise NotImplementedError

    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
