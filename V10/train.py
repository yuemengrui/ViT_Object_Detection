# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.data import DataLoader
from model import VisionTransformer
from dataset import ViTSegDataset
from metrics import SegMetrics
import logger as logger


class Trainer:
    def __init__(self, configs):
        self.train_loader = None
        self.train_data_total = 0
        self.train_loader_len = 0
        self.val_loader = None
        self.val_data_total = 0
        self.val_loader_len = 0
        self.configs = configs

        self.save_dir = self.configs.get('save_dir')
        self.best_model_save_path = os.path.join(self.save_dir, 'model_best.pth')
        if self.configs.get('device') == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')

        logger.basic.info('train with device {} and pytorch {}'.format(self.device, torch.__version__))

        self.global_step = 0
        self.start_epoch = 0
        self.epochs = self.configs.get('Epochs', 500)

        self._initialize()

        self.metrics = {'MeanIoU': 0, 'loss': float('inf'), 'best_model_epoch': 0}

    def train(self):
        """
        Full training logic
        """
        logger.train.info('start train')
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.epoch_result = self._train_epoch(epoch)
            self._on_epoch_finish()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        train_loss = 0.0

        batch_start = time.time()
        for i, (img, target, label) in enumerate(self.train_loader):
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            img = img.to(self.device)
            target = target.to(self.device)
            label = label.to(self.device)

            reader_cost = time.time() - batch_start

            cur_batch_size = img.size()[0]

            self.optimizer.zero_grad()
            preds = self.model(img, target)
            loss = self.criterion(preds, label.long())
            # backward
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            batch_cost = time.time() - batch_start
            if i % 50 == 0:
                logger.train.info(
                    '[epoch:{}/{}] [iter:{}/{}] global_step:{}, loss:{}, lr:{:.9f}, reader_cost:{:.2f}s, batch_cost:{:.2f}s, speed:{:.1f}/s'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step, loss.item(), lr,
                        reader_cost,
                        batch_cost, cur_batch_size / batch_cost))

            batch_start = time.time()

        return {'loss': train_loss / self.train_loader_len, 'epoch_cost': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self):
        logger.val.info('start eval, eval data total:{}'.format(self.val_data_total))
        self.model.eval()
        self.metric_cls.reset()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        total_frame = 0.0
        eval_start = time.time()
        # batch_start = time.time()
        for img, target, label in self.val_loader:
            with torch.no_grad():
                img = img.to(self.device)
                target = target.to(self.device)
                label = label.to(self.device)

                preds = self.model(img, target)

                preds = preds.data.cpu().numpy()
                label = label.cpu().numpy()
                pred = np.argmax(preds, axis=1)

                self.metric_cls.update(label, pred)
                # batch_cost = time.time() - batch_start
                total_frame += img.size()[0]
                # batch_start = time.time()

        metrics = self.metric_cls.get_results()

        logger.val.info(
            'eval finished. {} FPS:{:.2f}'.format(str(metrics), total_frame / (time.time() - eval_start)))
        return metrics, time.time() - eval_start

    def _on_epoch_finish(self):
        self.lr_scheduler.step()
        logger.train.info('epoch:{} finished. loss:{:.4f}, epoch_cost:{:.2f}s\n'.format(
            self.epoch_result['epoch'], self.epoch_result['loss'], self.epoch_result['epoch_cost']))

        metrics, eval_cost = self._eval()

        if metrics['MeanIoU'] >= self.metrics['MeanIoU']:
            self.metrics['loss'] = self.epoch_result['loss']
            self.metrics.update(metrics)
            self.metrics['best_model_epoch'] = self.epoch_result['epoch']

            self._save_checkpoint(self.epoch_result['epoch'])
            logger.val.info("Saving best model")

        # if self.epoch_result['epoch'] % 100 == 0:
        #     checkpoint_path = os.path.join(self.save_dir, 'epoch_{}.pth'.format(self.epoch_result['epoch']))
        #     self._save_checkpoint(self.epoch_result['epoch'], path=checkpoint_path)

        logger.val.info(
            'best model: {}\n'.format(str(self.metrics)))

    def _on_train_finish(self):

        logger.basic.info('train finished!')

    def _save_checkpoint(self, epoch, path=None):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best_old.pth.tar'
        """
        state_dict = self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict(),
            'configs': self.configs,
            'metrics': self.metrics
        }

        if path:
            torch.save(state, path)
        else:
            torch.save(state, self.best_model_save_path)

    def _load_checkpoint(self, checkpoint_path):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        logger.basic.info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])

        self.global_step = checkpoint['global_step']
        self.start_epoch = checkpoint['epoch']
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'metrics' in checkpoint:
            self.metrics = checkpoint['metrics']

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        logger.basic.info("resume from checkpoint (epoch {})".format(self.start_epoch))

        if self.start_epoch >= self.epochs:
            self.epochs += self.start_epoch

    def _initialize(self):
        start = time.time()

        self.model = VisionTransformer()

        # self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()
        # self.criterion = nn.L1Loss()
        weight = torch.FloatTensor([1.0, 100.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # self.criterion = FocalLoss()

        # self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.configs.get('lr'), momentum=0.9, dampening=0,
        #                            weight_decay=1e-4)

        self.optimizer = optim.AdamW(params=self.model.parameters(), lr=self.configs.get('lr'),
                                     weight_decay=self.configs.get('weight_decay'))

        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=500)
        # self.lr_scheduler = CosineLRScheduler(optimizer=self.optimizer, t_initial=self.configs.get('t_initial'),
        #                                       lr_min=5e-6, warmup_lr_init=5e-7,
        #                                       warmup_t=self.configs.get('warmup_t'), cycle_limit=1, t_in_epochs=False, )

        # self.lr_scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=50, gamma=0.9)

        # self.metric_cls = CornerMetric()
        self.metric_cls = SegMetrics()

        resume_checkpoint = self.configs.get('Train', {}).get('resume_checkpoint', '')
        if resume_checkpoint != '' and os.path.exists(resume_checkpoint):
            self._load_checkpoint(resume_checkpoint)
        self.model.to(self.device)

        t = time.time()
        logger.basic.info('build model finished. time_cost:{:.2f}s\n'.format(t - start))
        logger.basic.info('start load dataset')

        train_dataset = ViTSegDataset(self.configs.get('dataset_dir'))

        val_dataset = ViTSegDataset(self.configs.get('dataset_dir'), mode='val')

        self.train_data_total = len(train_dataset)
        self.val_data_total = len(val_dataset)

        self.train_loader = DataLoader(train_dataset, batch_size=self.configs.get('batch_size'), shuffle=True,
                                       drop_last=True)
        self.train_loader_len = len(self.train_loader)

        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
        self.val_loader_len = len(self.val_loader)

        datasrt_msg = '\n---------------Dataset Information---------------\n'
        datasrt_msg += 'train data total:{}'.format(self.train_data_total)
        datasrt_msg += '\nval data total:{}'.format(self.val_data_total)
        datasrt_msg += '\ntime_cost:{:.2f}s'.format(time.time() - t)
        datasrt_msg += '\ndataset load success'
        datasrt_msg += '\n---------------Dataset Information---------------\n'
        logger.basic.info(datasrt_msg)


if __name__ == '__main__':
    configs = {
        'save_dir': './checkpoints',
        'dataset_dir': '/data/guorui/ViT_DET/train_data_prebox',
        'Epochs': 10000,
        'batch_size': 64,
        'lr': 1e-4,
        'weight_decay': 1e-2,
        # 't_initial': 100000,  # EPOCHS * n_iter_per_epoch
        # 'warmup_t': 20*12,  # WARMUP_EPOCHS * n_iter_per_epoch
        'Train': {
            'resume_checkpoint': ''
        }
    }
    trainer = Trainer(configs)
    trainer.train()
