import argparse
import easydict

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-sampler_batch_size', default=256, help='Batch size to use in the box sampler', type=int)
    parser.add_argument('-num_pos', default=0, help='Number of positive examples', type=int)
    parser.add_argument('-sampler_high_thresh', default=0.75, help='Boxes with IoU greater than this with a GT box are considered positives', type=float)
    parser.add_argument('-sampler_low_thresh', default=0.4, help='Boxes with IoU less than this with all GT boxes are considered negatives', type=float)
    parser.add_argument('-train_remove_outbounds_boxes', default=1, help='Whether to ignore out-of-bounds boxes for sampling at training time', type=int)
    
    # Model 
    parser.add_argument('-embedding', default='dct', help='Which embedding to use, dct or phoc')
    parser.add_argument('-embedding_loss', default='cosine', help='which embedding loss to use')
    parser.add_argument('-num_layers', default=34, help='Number of resnet layers', type=int)
    parser.add_argument('-wordness_loss', default='cross_entropy', help='Which loss to use for wordness scores')
    parser.add_argument('-std', default=0.01, help='std for init', type=int)
    parser.add_argument('-init_weights', default=1, help='unit normal weight initialization', type=int)
    parser.add_argument('-embedding_net_layers', default=2, help='Number of hidden layers for embedding net', type=int)
    parser.add_argument('-emb_fc_size', default=4096, help='embedding net hidden size', type=int)
    
    # Loss function
    parser.add_argument('-mid_box_reg_weight', default=0.01, help='Weight for box regression in the RPN', type=float)
    parser.add_argument('-mid_objectness_weight', default=0.01, help='Weight for box classification in the RPN', type=float)
    parser.add_argument('-end_box_reg_weight', default=0.1, help='Weight for box regression in the recognition network', type=float)
    parser.add_argument('-end_objectness_weight', default=0.1, help='Weight for box classification in the recognition network', type=float)
    parser.add_argument('-embedding_weight', default=3.0, help='Weight for embedding loss', type=float)
    parser.add_argument('-weight_decay', default=1e-5, help='L2 weight decay penalty strength', type=float)
    parser.add_argument('-box_reg_decay', default=5e-5, help='Strength of pull that boxes experience towards their anchor', type=float)
    parser.add_argument('-cosine_margin', default=0.1, help='margin for the cosine embedding loss', type=float)
    
    # Data input settings
    parser.add_argument('-dataset', default='washington', help='Which dataset to train on')
    parser.add_argument('-val_dataset', default='', help='Which dataset to use as validation and testing')
    parser.add_argument('-fold', default=1, help='which fold to use', type=int)
    parser.add_argument('-image_size', default=1720, help='which fold to use', type=int)
    parser.add_argument('-dtp_train', default=0, help='use dtp while training', type=int)
    parser.add_argument('-num_workers', default=6, help='number of data loader workers', type=int)
    parser.add_argument('-augment', default=1, help='use augmentation', type=int)
    parser.add_argument('-aug_ratio', default=0.5, help='proportion of inplace to full, 0.5=equal', type=float)
    parser.add_argument('-augment_mode', default='classic', help='augment mode')
    parser.add_argument('-h5', default=0, help='to load h5 datasets', type=int)
    
    # Optimization
    parser.add_argument('-learning_rate', default=1e-3, help='learning rate to use', type=float)
    parser.add_argument('-reduce_lr_every', default=10000, help='reduce learning rate every x iterations', type=int)
    parser.add_argument('-beta1', default=0.9, help='beta1 for adam', type=float)
    parser.add_argument('-beta2', default=0.999, help='beta2 for adam', type=float)
    parser.add_argument('-epsilon', default=1e-8, help='epsilon for smoothing', type=float)
    parser.add_argument('-max_iters', default=25000, help='Number of iterations to run; -1 to run forever', type=int)
    parser.add_argument('-weights', default='', help='Load model from a checkpoint instead of random initialization.')
    
    # Model checkpointing
    parser.add_argument('-eval_every', default=1000, help='How often to test on validation set', type=int)
    
    # Test-time model options (for evaluation)
    parser.add_argument('-test_rpn_nms_thresh', default=0.4, help='Test-time NMS threshold to use in the RPN', type=float)
    parser.add_argument('-max_proposals', default=-1, help='Number of region proposal to use at test-time', type=int)
    parser.add_argument('-query_nms_overlap', default=0.0, help='NMS overlap during querying', type=float)
    parser.add_argument('-score_nms_overlap', default=0.4, help='NMS overlap using box scores', type=float)
    parser.add_argument('-score_threshold', default=0.01, help='threshold using box scores', type=float)
    parser.add_argument('-external_proposals', default=1, help='Whether or not to use DTP', type=int)
    parser.add_argument('-test_batch_size', default=128, help='Whether or not to use DTP', type=int)
    parser.add_argument('-folds', default=0, help='multiple folds for testing', type=int)
    parser.add_argument('-save', default=0, help='save test results to json', type=int)
    parser.add_argument('-hyperparam_opt', default=0, help='optimize hyperparams before testing', type=int)
    parser.add_argument('-nms_max_boxes', default=None, help='max number of boxes to keep for nms', type=int)
    parser.add_argument('-reproduce_paper', default=0, help='use exact data from paper', type=int)

    # Visualization
    parser.add_argument('-print_every', default=200, help='How often to print the latest images training loss.', type=int)
    
    # Misc
    parser.add_argument('-save_id', default='', help='an id identifying this run/job')
    parser.add_argument('-quiet', default=0, help='run in quiet mode, no prints', type=int)
    parser.add_argument('-verbose', default=0, help='print trianing info', type=int)
    parser.add_argument('-gpu', default=0, help='which gpu to use.', type=int)
    parser.add_argument('-seed', default=123, help='which gpu to use.', type=int)
    parser.add_argument('-clip_final_boxes', default=1, help='Whether to clip final boxes to image boundar', type=int)
    parser.add_argument('-eval_first_iteration', default=1, help='evaluate on first iteration? 1 = do, 0 = dont.', type=int)
    parser.add_argument('-dtp_only', default=0, help='test using dtp only model.', type=int)
    parser.add_argument('-ghosh', default=0, help='test using ghosh evaluation.', type=int)
    
    args = parser.parse_args()
    
    if args.val_dataset == '':
        args.val_dataset = args.dataset
        
    if args.reproduce_paper:
        args.h5 = 1
        
    return easydict.EasyDict(vars(args))
