import torch
from torch import optim
from timm.scheduler import CosineLRScheduler


def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
            
        optimizer = optim.AdamW(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=False, **opti_config.kwargs)
    else:
        raise NotImplementedError()
    sche_config = config.scheduler
    if sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                lr_min=sche_config.kwargs.lr_min,
                cycle_decay=0.1,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    elif sche_config.type == "Plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                patience=sche_config.patience,
                min_lr= 1e-6,
                factor=0.1)
    else:
        raise NotImplementedError()
    
    return optimizer, scheduler
