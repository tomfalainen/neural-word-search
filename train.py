import warnings
warnings.filterwarnings("ignore")
import os
import json
import easydict
import time
import torch
from misc.dataloader import DataLoader
import torch.optim as optim
import misc.datasets as datasets
import ctrlfnet_model as ctrlf
from train_opts import parse_args
from evaluate import mAP
import misc.h5_dataset as h5_dataset

opt = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

if opt.h5:
    trainset = h5_dataset.H5Dataset(opt, split=0)
    valset = h5_dataset.H5Dataset(opt, split=1)
    testset = h5_dataset.H5Dataset(opt, split=2)
    opt.num_workers = 0
else:
    if opt.dataset.find('iiit_hws') > -1:
        trainset = datasets.SegmentedDataset(opt, 'train')
    else:
        trainset = datasets.Dataset(opt, 'train')

    valset = datasets.Dataset(opt, 'val')
    testset = datasets.Dataset(opt, 'test')
sampler=datasets.RandomSampler(trainset, opt.max_iters)
trainloader = DataLoader(trainset, batch_size=1, sampler=sampler, num_workers=opt.num_workers)
valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)


torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.device(opt.gpu)

# initialize the Ctrl-F-Net model object
model = ctrlf.CtrlFNet(opt)

show = not opt.quiet
if show:
    print "number of parameters in ctrlfnet:", model.num_parameters()

model.load_weights(opt.weights)
model.cuda()

optimizer = optim.Adam(model.parameters(), opt.learning_rate, (opt.beta1, opt.beta2),opt.epsilon, opt.weight_decay)
keys = ['bd', 'e', 'ebr', 'eo', 'mbr', 'mo', 'total_loss']
running_losses = {k:0.0 for k in keys}

it = 0
args = easydict.EasyDict()
args.nms_overlap = opt.query_nms_overlap
args.score_threshold = opt.score_threshold
args.num_queries = -1
args.score_nms_overlap = opt.score_nms_overlap
args.overlap_threshold = 0.5
args.gpu = True
args.use_external_proposals = int(opt.external_proposals)
args.max_proposals = opt.max_proposals
args.rpn_nms_thresh = opt.test_rpn_nms_thresh
args.num_workers = 6
args.numpy = False
args.num_workers = 6

trainlog = ''
start = time.time()
loss_history, mAPs = [], []
if opt.eval_first_iteration:
    log, rf, rt = mAP(model, valloader, args, it)
    trainlog += log
    if show:
        print(log)
    best_score = (rt.mAP_qbe_50 + rt.mAP_qbs_50) / 2
    mAPs.append((it, [rt.mAP_qbe_50, rt.mAP_qbs_50]))
    
else:
    best_score = 0.0

if opt.weights:
    opt.save_id += '_pretrained'
    
if not os.path.exists('checkpoints/ctrlfnet/'):
    os.makedirs('checkpoints/ctrlfnet/')
    
oargs = ('ctrlfnet', opt.embedding, opt.dataset, opt.fold, opt.save_id)
out_name = 'checkpoints/%s/%s_%s_fold%d_%s_best_val.pt' % oargs
for data in trainloader:
    optimizer.zero_grad()
    losses = model.forward_backward(data, True)
    optimizer.step()
    
    # print statistics
    running_losses  = {k:v + losses[k] for k, v in running_losses.iteritems()}
    if it % opt.print_every == opt.print_every - 1:
        running_losses  = {k:v / opt.print_every for k, v in running_losses.iteritems()}
        loss_string = "[iter %5d] " % (it + 1)
        for k, v in running_losses.iteritems():
            loss_string += "%s: %.5f | " % (k , v)
            
        trainlog += loss_string
        if show:
            print loss_string
        vals = [val[0] for val in running_losses.values()]
        loss_history.append((it, vals))
        running_losses  = {k:0.0 for k, v in running_losses.iteritems()}
        
    if it % opt.eval_every == opt.eval_every - 1:
        log, rf, rt = mAP(model, valloader, args, it)
        trainlog += log
        if show:
            print(log)
        score = (rt.mAP_qbe_50 + rt.mAP_qbs_50) / 2
        mAPs.append((it, [rt.mAP_qbe_50, rt.mAP_qbs_50]))
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), out_name)
            if show:
                print  'saving ' + out_name
            
        d = {}
        d['opt'] = opt
        d['loss_history'] = loss_history
        d['map_history'] = mAPs
        d['trainlog'] = trainlog
        with open(out_name + '.json', 'w') as f:
            json.dump(d, f)

    if it % opt.reduce_lr_every == opt.reduce_lr_every - 1:
        optimizer.param_groups[0]['lr'] /= 10.0
        
    it += 1

if show:
    if opt.val_dataset != 'iam':
        model.load_weights(out_name)
        log, rf, rt = mAP(model, testloader, args, it)
        print(log)
        
    d = {}
    d['opt'] = opt
    d['loss_history'] = loss_history
    d['map_history'] = mAPs
    d['trainlog'] = trainlog
    d['testlog'] = log
    with open(out_name + '.json', 'w') as f:
        json.dump(d, f)
    
    duration = time.time() - start
    print "training model took %0.2f hours" % (duration / 3600)
