import torch
import torch.nn as nn
import torch.nn.functional as F


logit = torch.randn(10, 4)
target = torch.randint(4, (10,))


# =============================================================================
# 1. CrossEntropy Loss
loss_fn = nn.CrossEntropyLoss()
loss1 = loss_fn(logit, target)
# ----------------------------------------------------------------------
# 2. NLL Loss
log_softmax_logit = F.log_softmax(logit)
loss_fn2 = nn.NLLLoss()
loss2 = loss_fn2(log_softmax_logit, target)
# ----------------------------------------------------------------------
# 3. Manually
logit1 = F.softmax(logit)       # softmax version
logit2 = torch.log(logit1)      # log of softmax version
logit3 = (-1) * logit2          # negative log of softmax version.
selected = torch.gather(logit3, dim=1, index=target.unsqueeze(1))
loss3 = torch.mean(selected)
print()


