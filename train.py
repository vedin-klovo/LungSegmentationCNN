import torch
import torch.nn as nn
import logging

from typing import Tuple
from enum import IntEnum
from tqdm import tqdm

from torch import optim
import torch.nn.functional as F

from unet.unet import UNet

from lung_seg_dataset import MontgomeryDataset, ShenzhenDataset
from torch.utils.data import DataLoader, random_split
import wandb

dir_checkpoints = './checkpoints'


class DatasetChoice(IntEnum):
    SHENZHEN = 0,
    MONTGOMERY = 1,


def train_net(
        net: nn,
        device: torch.device,
        dataset_choice: DatasetChoice = DatasetChoice.SHENZHEN,
        epochs: int = 32,
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        train_percent: float = 0.9,
        save_checkpoint: bool = False,
        input_size: Tuple[int, int] = (572, 572),
):
    if dataset_choice == DatasetChoice.SHENZHEN:
        dataset = ShenzhenDataset()
    elif dataset_choice == DatasetChoice.MONTGOMERY:
        dataset = MontgomeryDataset()
    else:
        raise Exception("Wrong dataset choice.")

    n_train = int(len(dataset) * train_percent)
    n_val = int(len(dataset) - n_train)

    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(project='MasterRad', resume='allow', anonymous='must')
    experiment.config.update(dict(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_percent=train_percent,
        save_checkpoint=save_checkpoint,
        input_size=input_size
    ))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Input size:      {input_size}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(True):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks)
                    loss.backward()
                    optimizer.step()
                           #+ dice_loss(F.softmax(masks_pred, dim=1).float(),
                           #            F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                           #            multiclass=True)

                # optimizer.zero_grad(set_to_none=True)
                #grad_scaler.scale(loss).backward()
                #grad_scaler.step(optimizer)
                #grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = 0  # evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint:
            pass
            # Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            # logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    net = UNet(input_channel=1, num_classes=2)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 )

    net.to(device)

    try:
        train_net(net, device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
