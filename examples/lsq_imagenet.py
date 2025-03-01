import time
import warnings

warnings.filterwarnings("ignore")
from loguru import logger
import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.dataset import Subset

from brocolli.quantization.quantizer import PytorchQuantizer
from brocolli.testing.dataset import ImageNetDatasetCHINA
from brocolli.testing.quant_utils import AverageMeter, ProgressMeter, accuracy


def calibrate_func(model):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = ImageNetDatasetCHINA(
        "data",
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # compute output
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        logger.info(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")


model = models.resnet18(pretrained=True)
model.eval()

pytorch_quantizer = PytorchQuantizer(model, (1, 3, 224, 224))
pytorch_quantizer.fuse()
pytorch_quantizer.prepare_calibration()
pytorch_quantizer.calibrate(calibrate_func)
pytorch_quantizer.prepare_lsq()
pytorch_quantizer.lsq(calibrate_func)
pytorch_quantizer.export_graph('resnet18.json', 'resnet18.onnx')
