from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import ArcFaceLoss1, ArcFaceLoss2, CrossEntropyLoss
import torch 

from ..engine import Engine


class ImageArcfaceEngine(Engine):
    r"""Arcface-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of ``torchreid.data.ImageDataManager``
            or ``torchreid.data.VideoDataManager``.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        scheduler (LRScheduler, optional): if None, no learning rate decay will be performed.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer. Default is True.

    Examples::
        
        import torchreid
        datamanager = torchreid.data.ImageDataManager(
            root='path/to/reid-data',
            sources='market1501',
            height=256,
            width=128,
            combineall=False,
            batch_size=32
        )
        model = torchreid.models.build_model(
            name='resnet50',
            num_classes=datamanager.num_train_pids,
            loss='arcface'
        )
        model = model.cuda()
        optimizer = torchreid.optim.build_optimizer(
            model, optim='adam', lr=0.0003
        )
        scheduler = torchreid.optim.build_lr_scheduler(
            optimizer,
            lr_scheduler='single_step',
            stepsize=20
        )
        engine = torchreid.engine.ImageArcfaceEngine(
            datamanager, model, optimizer, scheduler=scheduler
        )
        engine.run(
            max_epoch=60,
            save_dir='log/resnet50-arcface-market1501',
            print_freq=10
        )
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        scale=64,
        margin=0.5,
        easy_margin=False,
        use_gpu=True,
        label_smooth=True
    ):
        super(ImageArcfaceEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)
        
        self.criterion_t = ArcFaceLoss1(
            embed_size=self.model.module.feature_dim,
            num_classes=self.datamanager.num_train_pids,
            scale=scale, margin=margin, easy_margin=easy_margin, label_smooth=label_smooth,
            use_gpu=self.use_gpu,
        )
        
        # self.criterion_t = ArcFaceLoss2(
        #     embed_size=self.model.module.feature_dim,
        #     num_classes=self.datamanager.num_train_pids,
        #     use_gpu=self.use_gpu,
        # )
        
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        loss = 0
        outputs, emds = self.model(imgs)
        # print(f"compute_loss imgs shape: {imgs.size()}")
        # print(f"compute_loss pids shape: {pids.size()}")
        # print(f"compute_loss output shape: {outputs.size()}")
        loss_t = self.compute_loss(self.criterion_t, emds, pids)
        loss_x = self.compute_loss(self.criterion_x, outputs, pids)
        
        loss = loss_t * 0.5 + loss_x
        # loss = res['loss']
        # outputs = res['probs']
        
        # """ probs [B, Numclasses] """
        # print(probs.size())
        # print(probs)
        # max_probs, outputs = torch.max(probs, dim=1)
        # print(max_probs)
        # print(outputs)
        # print(pids)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss_t': loss_t.item(),
            'loss_x': loss_x.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary
