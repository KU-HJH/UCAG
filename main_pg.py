import os

from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset

from datasets import *
from pointing_game import *
from torchvision import models, transforms
from cam import *
from tqdm import tqdm
from demo_eval import multi_scale_fusion
from models import get_model, get_transform

series = 'attribution_benchmarks'
series_dir = os.path.join('data', series)
log = 0
seed = 0
chunk = None

from parsing import get_cam, parsing

datasets = [
    'voc_2007',
    'coco'
]

archs = [
    'vgg16',
    'resnet50'
]


methods = [
    'grad_cam',
]

args = parsing()


class ProcessingError(Exception):
    def __init__(self, executor, experiment, model, image, label, class_id, image_size):
        super().__init__(f"Error processing {str(label):20s}")
        self.executor = executor
        self.experiment = experiment
        self.model = model
        self.image = image
        self.label = label
        self.class_id = class_id
        self.image_size = image_size


def _saliency_to_point(saliency):
    assert len(saliency.shape) == 4
    w = saliency.shape[3]
    point = torch.argmax(
        saliency.view(len(saliency), -1),
        dim=1,
        keepdim=True
    )
    return torch.cat((point % w, point // w), dim=1)

def get_saliency(cam, input, target, image_size, mode='original'):

    assert mode in ['original', 'spatial_integral']
    # batch size = 1

    if not args.ours:
        if args.image_size != 224:
            return cam(torch.nn.functional.interpolate(input, scale_factor=(args.alpha, args.alpha), mode='bicubic', align_corners=False), class_idx=target, image_size=image_size[::-1])
        return cam(input, class_idx=target, image_size=image_size[::-1])
    else:
        return multi_scale_fusion(cam, input, target, args=args, image_size=image_size[::-1])
    # else:
    #     assert False

class ExperimentExecutor():

    def __init__(self, experiment, chunk=None, debug=0, log=0, seed=seed):
        self.experiment = experiment
        self.device = None
        self.model = None
        self.data = None
        self.loader = None
        self.pointing = None
        self.pointing_difficult = None
        self.debug = debug
        self.log = log
        self.seed = seed

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = get_model(
            arch=self.experiment.arch,
            dataset=self.experiment.dataset,
            convert_to_fully_convolutional=False,
        )

        if self.experiment.arch == 'vgg16':
            self.gradcam_layer = 'features.29'  # relu before pool5
            self.saliency_layer = 'features.23'  # pool4
            self.contrast_layer = 'classifier.4'  # relu7
        elif self.experiment.arch == 'resnet50':
            self.gradcam_layer = 'layer4'
            self.saliency_layer = 'layer3'  # 'layer3.5'  # res4a
            self.contrast_layer = 'avgpool'  # pool before fc layer
        else:
            assert False
        print(self.experiment.arch)
        if self.experiment.arch == 'resnet50':
            if any([e in self.experiment.method for e in [
                    'contrastive_excitation_backprop',
                    'deconvnet',
                    'excitation_backprop',
                    'grad_cam',
                    'gradient',
                    'guided_backprop'
            ]]):
                self.model.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                # self.model.avgpool = torch.nn.AvgPool2d((7, 7), stride=1)
        
        self.model.eval()
        self.model.cuda()
        
        # self.cam = GradCAM(self.model, target_layer=self.gradcam_layer)
        # self.cam = CAMERAS(self.model, target_layer=self.gradcam_layer)
        self.cam = get_cam(args.cam, self.model, self.gradcam_layer)

        if self.experiment.dataset == 'voc_2007':
            subset = 'test'
        elif self.experiment.dataset == 'coco':
            subset = 'val2014'
        else:
            assert False

        # Load the model.
        if self.experiment.method == "rise" or self.experiment.method == "gradcam":
            input_size = (224, 224)
        else:
            input_size = 224
        transform = get_transform(size=input_size,
                                  dataset=self.experiment.dataset)
        self.data = get_dataset(name=self.experiment.dataset,
                                subset=subset,
                                transform=transform,
                                download=False,
                                limiter=None)

        # Get subset of data. This is used for debugging and for
        # splitting computation on a cluster.
        if chunk is None:
            chunk = self.experiment.chunk

        if isinstance(chunk, dict):
            dataset_filter = chunk
            chunk = []
            if 'image_name' in dataset_filter:
                for i, name in enumerate(self.data.images):
                    if dataset_filter['image_name'] in name:
                        chunk.append(i)

            print(f"Filter selected {len(chunk)} image(s).")

         # Limit the chunk to the actual size of the dataset.
        if chunk is not None:
            chunk = list(set(range(len(self.data))).intersection(set(chunk)))

        # Extract the data subset.
        chunk = Subset(self.data, chunk) if chunk is not None else self.data

        # Get a data loader for the subset of data just selected.
        self.loader = DataLoader(chunk,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=32,
                                 collate_fn=self.data.collate)

        self.pointing = PointingGameBenchmark(self.data, difficult=False)
        self.pointing_difficult = PointingGameBenchmark(
            self.data, difficult=True)
        self.energy_pointing = EnergyPointingGameBenchmark(self.data, difficult=False)
        self.energy_pointing_difficult = EnergyPointingGameBenchmark(
            self.data, difficult=True)

        self.data_iterator = iter(self.loader)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader.dataset)

    def __next__(self):
        x, y = next(self.data_iterator)
        torch.manual_seed(self.seed)

        # try:
        assert len(x) == 1
        x = x.cuda()
        class_ids = self.data.as_class_ids(y[0])
        image_size = self.data.as_image_size(y[0])

        results = {
            'pointing': {},
            'pointing_difficult': {},
            'energy_pointing': {},
            'energy_pointing_difficult': {}
        }
        info = {}

        for class_id in class_ids:

            # TODO(av): should now be obsolete
            if x.grad is not None:
                x.grad.data.zero_()

            if self.experiment.method == "grad_cam":
                saliency = get_saliency(
                    self.cam, x, class_id, image_size
                )
                point = _saliency_to_point(saliency).cpu()
                info['saliency'] = saliency
            else:
                assert False

            if False:
                plt.figure()
                plt.subplot(1, 2, 1)
                imsc(saliency[0])
                plt.plot(point[0, 0], point[0, 1], 'ro')
                plt.subplot(1, 2, 2)
                imsc(x[0])
                plt.pause(0)

            results['pointing'][class_id] = self.pointing.evaluate(
                y[0], class_id, point[0].cpu())
            results['pointing_difficult'][class_id] = self.pointing_difficult.evaluate(
                y[0], class_id, point[0].cpu())
            results['energy_pointing'][class_id] = self.energy_pointing.evaluate(
                y[0], class_id, saliency)
            results['energy_pointing_difficult'][class_id] = self.energy_pointing_difficult.evaluate(
                y[0], class_id, saliency)

        return results

        # except Exception as ex:
        #     raise ProcessingError(
        #         self, self.experiment, self.model, x, y, class_id, image_size) from ex

    def aggregate(self, results):
        for class_id, hit in results['pointing'].items():
            self.pointing.aggregate(hit, class_id)
        for class_id, hit in results['pointing_difficult'].items():
            self.pointing_difficult.aggregate(hit, class_id)
        for class_id, hit in results['energy_pointing'].items():
            self.energy_pointing.aggregate(hit, class_id)
        for class_id, hit in results['energy_pointing_difficult'].items():
            self.energy_pointing_difficult.aggregate(hit, class_id)

    def run(self, save=True):
        all_results = []
        for itr, results in enumerate(tqdm(self)):
            all_results.append(results)
            self.aggregate(results)
            if itr % max(len(self) // 20, 1) == 0 or itr == len(self) - 1:
                print("[{}/{}]".format(itr + 1, len(self)))
                print(self)

        print("[final result]")
        print(self)

        self.experiment.pointing = self.pointing.accuracy
        self.experiment.pointing_difficult = self.pointing_difficult.accuracy
        self.experiment.energy_pointing = self.pointing.accuracy
        self.experiment.energy_pointing_difficult = self.pointing_difficult.accuracy
        if save:
            self.experiment.save()

        return all_results

    def __str__(self):
        return (
            f"{args.cam} {self.experiment.arch} "
            f"{self.experiment.dataset} "
            f"pointing_game: {self.pointing}\n"
            f"{args.cam} {self.experiment.arch} "
            f"{self.experiment.dataset} "
            f"pointing_game(difficult): {self.pointing_difficult}\n"
            f"{args.cam} {self.experiment.arch} "
            f"{self.experiment.dataset} "
            f"energy_pointing_game: {self.energy_pointing}\n"
            f"{args.cam} {self.experiment.arch} "
            f"{self.experiment.dataset} "
            f"energy_pointing_game_difficult: {self.energy_pointing_difficult}"
        )

class Experiment():
    def __init__(self,
                 series,
                 method,
                 arch,
                 dataset,
                 root='',
                 chunk=None,
                 boom=False):
        self.series = series
        self.root = root
        self.method = method
        self.arch = arch
        self.dataset = dataset
        self.chunk = chunk
        self.boom = boom
        self.pointing = float('NaN')
        self.pointing_difficult = float('NaN')
        self.energy_pointing = float('NaN')
        self.energy_pointing_difficult = float('NaN')

    def __str__(self):
        return (
            f"{self.method},{self.arch},{self.dataset},"
            f"{self.pointing:.5f},{self.pointing_difficult:.5f},"
            f"{self.energy_pointing:.5f},{self.energy_pointing_difficult:.5f}"
        )

    @property
    def name(self):
        return f"{self.method}-{self.arch}-{self.dataset}"

    @property
    def path(self):
        return os.path.join(self.root, self.name + ".csv")

    def save(self):
        with open(self.path, "w") as f:
            f.write(self.__str__() + "\n")

    def load(self):
        with open(self.path, "r") as f:
            data = f.read()
        method, arch, dataset, pointing, pointing_difficult, energy_pointing, energy_pointing_difficult = data.split(",")
        assert self.method == method
        assert self.arch == arch
        assert self.dataset == dataset
        self.pointing = float(pointing)
        self.pointing_difficult = float(pointing_difficult)
        self.energy_pointing = float(energy_pointing)
        self.energy_pointing_difficult = float(energy_pointing_difficult)

    def done(self):
        return os.path.exists(self.path)

experiments = []

for d in datasets:
    for a in archs:
        for m in methods:
            experiments.append(
                Experiment(series=series,
                           method=m,
                           arch=a,
                           dataset=d,
                           chunk=chunk,
                           root=series_dir))

if __name__ == "__main__":
    for e in experiments:
        if e.done():
            e.load()
            continue
        ExperimentExecutor(e, log=log).run()