import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='ZS-SSL: Zero-Shot Self-Supervised Learning')

    # Hyperparameters for the network
    parser.add_argument('--data_dir', type=str, default='data/brain_T1w.npy',
                        help='data directory') 
    parser.add_argument('--ncontrast', type=int, default=1,
                        help='number of contrasts of the slices in the dataset')  
    parser.add_argument('--ndepth', type=int, default=38,
                        help='number of slices in the dataset')            
    parser.add_argument('--nrow', type=int, default=150,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol', type=int, default=136,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil', type=int, default=1,
                        help='number of coils of the slices in the dataset')              
    parser.add_argument('--acc_rate', type=int, default=2,
                        help='acceleration rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default=5,
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default=5,
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter', type=int, default=10,
                        help='number of Conjugate Gradient iterations for DC')

    # Hyperparameters for the 3D-ZS-SSL
    parser.add_argument('--rho_val', type=float, default=0.2,
                        help='cardinality of the validation mask')                        
    parser.add_argument('--rho_train', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')
    parser.add_argument('--num_reps', type=int, default=25,
                        help='number of repetions for the remainder mask')
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='transfer learning from pretrained model')
    parser.add_argument('--TL_path', type=str, default="pretrained_weights/Pretraining_R2_5Unrolls_5ResNet/best.pth",
                        help='path to pretrained model')                                        
    parser.add_argument('--stop_training', type=int, default=5,
                        help='stop training if a new lowest validation loss hasnt been achieved in xx epochs')                         
    return parser
