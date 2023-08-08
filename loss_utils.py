import glob
import os.path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


def NSS_loss(input, saliency, target):  # This NSS loss is UNISAL paper's implementation, note that the NSS loss weight in their paper is -0.1
    pred, fixations=input, target
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    fixations = fixations.reshape(new_size)

    pred_normed = (pred - pred.mean(-1, True)) / pred.std(-1, keepdim=True)
    results = []
    for this_pred_normed, mask in zip(torch.unbind(pred_normed, 0),
                                      torch.unbind(fixations, 0)):
        if mask.sum() == 0:
            print("No fixations.")
            results.append(torch.ones([]).float().to(fixations.device))
            continue
        nss_ = torch.masked_select(this_pred_normed, mask)
        nss_ = nss_.mean(-1)
        results.append(nss_)
    results = torch.stack(results)
    results = results.reshape(size[:2])
    return results

def NSS_loss_v1(input, saliency, target):  # This NSS loss is EMLNet paper's implementation,, note that the NSS loss weight in their paper is 1
    input=input.float()
    saliency=saliency.float()
    target=target.float()
    ref = (target - target.mean()) / target.std()
    input = (input - input.mean()) / input.std()
    loss = (ref*target - input*target).sum(-1).sum(-1).sum(-1) / target.sum(-1).sum(-1).sum(-1)
    return loss

def SFNE_loss_v1(input, saliency, target):  # This SFNE loss (based on NSS loss) is our implementation v1
    input=input.float()
    saliency=saliency.float()
    target=target.float()
    ref = (saliency - saliency.mean()) / saliency.std()
    input = (input - input.mean()) / input.std()
    #loss = torch.abs(ref*target - input*target).sum(-1).sum(-1).sum(-1) / target.sum(-1).sum(-1).sum(-1)
    loss = torch.pow((ref*target - input*target),2).sum(-1).sum(-1).sum(-1) / target.sum(-1).sum(-1).sum(-1)
    return loss

def SFNE_loss(input, saliency, target, nontarget):  # This SFNE loss (based on NSS loss) is our implementation
    input=input.float()
    saliency=saliency.float()
    target=target.float()
    nontarget=nontarget.float()

    ref = (saliency - saliency.mean()) / saliency.std()
    input = (input - input.mean()) / input.std()
    #loss = torch.abs(ref*target - input*target).sum(-1).sum(-1).sum(-1) / target.sum(-1).sum(-1).sum(-1)
    loss1 = torch.pow((ref*target - input*target),2).sum(-1).sum(-1).sum(-1) / target.sum(-1).sum(-1).sum(-1) # salient points
    loss2 = torch.pow((ref*nontarget - input*nontarget),2).sum(-1).sum(-1).sum(-1) / nontarget.sum(-1).sum(-1).sum(-1) # nonsalient points
    loss=loss1+loss2
    return loss

def NSS_loss_v2(input, saliency, target):  # This NSS loss is our SFNE_loss implementation plus unisal NSS_loss1 implementation
    loss_nss1=NSS_loss(input, saliency, target)
    loss_nss2=SFNE_loss(input, saliency, target)
    loss=(-1)*loss_nss1+loss_nss2
    return loss


def corr_coeff(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        r_num = torch.mean(xm * ym)
        r_den = torch.sqrt(
            torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
        r = r_num / r_den
        cc.append(r)

    cc = torch.stack(cc)
    cc = cc.reshape(size[:2])
    #return cc  # 1 - torch.square(r), note that the Unisal paper use (-0.1)*cc as cc loss 
    return 1-cc   # here we use 1-cc as cc loss 

def corr_coeff_v2(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        r_num = torch.mean(xm * ym)
        r_den = torch.sqrt(
            torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
        r = r_num / r_den
        cc.append(r)

    cc = torch.stack(cc)
    cc = cc.reshape(size[:2])
    return cc  # 1 - torch.square(r), note that the Unisal paper use (-0.1)*cc as cc loss 


def kld_loss(pred, target):
    loss = F.kl_div(pred, target, reduction='none')
    loss = loss.sum(-1).sum(-1).sum(-1)
    return loss

def loss_sequences(pred_seq, sal_seq, fix_seq, nonfix, metrics):
    """
    Compute the training losses
    """

    # sal_seq = softmax(sal_seq)
    # pred_seq = log_softmax(pred_seq)

    losses = []
    for this_metric in metrics:
        if this_metric == 'kld':
            losses.append(kld_loss(log_softmax(pred_seq), softmax(sal_seq)))  
        if this_metric == 'nss':
            losses.append(NSS_loss(pred_seq, sal_seq, fix_seq))  
        if this_metric == 'cc':
            losses.append(corr_coeff(pred_seq, sal_seq))
        if this_metric == 'sfne':
            losses.append(SFNE_loss(pred_seq, sal_seq, fix_seq, nonfix))
    return losses

def loss_sequences_val(pred_seq, sal_seq, fix_seq, nonfix, metrics):
    """
    Compute the training losses
    """

    # sal_seq = softmax(sal_seq)
    # pred_seq = log_softmax(pred_seq)

    losses = []
    for this_metric in metrics:
        if this_metric == 'kld':
            losses.append(kld_loss(log_softmax(pred_seq), softmax(sal_seq)))  # real KLD metric
        if this_metric == 'nss':
            losses.append(NSS_loss(pred_seq, sal_seq, fix_seq))    # real NSS metric
        if this_metric == 'cc':
            losses.append(corr_coeff_v2(pred_seq, sal_seq))     # real CC metric
    return losses

def softmax(x):
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.softmax(x, dim=1)
    return x.view(x_size)

def log_softmax(x):
    x_size = x.size()
    x = x.view(x.size(0), -1)
    x = F.log_softmax(x, dim=1)
    return x.view(x_size)

def calc_loss_unisal(pred_seq, sal, fix, nonfix, loss_metrics=('kld', 'nss', 'cc', 'sfne'), loss_weights=(1.0, -1.0, 1.0, 1.0)):
    assert len(pred_seq.shape) == 4 and len(sal.shape) == 4 and len(fix.shape) == 4
    sal = torch.unsqueeze(sal, 1)  # [b, nframe, c, w, h]
    fix = torch.unsqueeze(fix, 1)  # [b, nframe, c, w, h]
    nonfix = torch.unsqueeze(nonfix, 1)  # [b, nframe, c, w, h]
    pred_seq = torch.unsqueeze(pred_seq, 1)  #[b, nframe, c, w, h]

    # Compute the total loss
    loss_summands = loss_sequences(
        pred_seq, sal, fix, nonfix, metrics=loss_metrics)
    loss_summands = [l.mean(1).mean(0) for l in loss_summands]
    loss = sum(weight * l for weight, l in
               zip(loss_weights, loss_summands))
    return loss

def calc_loss_unisal_val(pred_seq, sal, fix, nonfix, loss_metrics=('kld', 'nss', 'cc'), loss_weights=(-1.0, 1.0, 1.0)):  # real KLD, NSS, CC metrics for validation
    assert len(pred_seq.shape) == 4 and len(sal.shape) == 4 and len(fix.shape) == 4
    sal = torch.unsqueeze(sal, 1)  # [b, nframe, c, w, h]
    fix = torch.unsqueeze(fix, 1)  # [b, nframe, c, w, h]
    nonfix = torch.unsqueeze(nonfix, 1)  # [b, nframe, c, w, h]
    pred_seq = torch.unsqueeze(pred_seq, 1)  #[b, nframe, c, w, h]

    # Compute the total loss
    loss_summands = loss_sequences_val(
        pred_seq, sal, fix, nonfix, metrics=loss_metrics)
    loss_summands = [l.mean(1).mean(0) for l in loss_summands]
    loss = sum(weight * l for weight, l in
               zip(loss_weights, loss_summands))
    return loss

