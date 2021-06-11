from itertools import islice
from tqdm import tqdm
import time

from KD_Lib.KD.common import BaseClass


class Pipeline:
    """
    Pipeline of knowledge distillation, pruning and quantization methods
    supported by KD_Lib. Sequentially applies a list of methods on the student model.

    All the elements in list must implement either train_student, prune or quantize
    methods.

    :param: steps (list) list of KD_Lib.KD or KD_Lib.Pruning or KD_Lib.Quantization
    :param: epochs (int) number of iterations through whole batch for each method in
                         list
    :param: plot_losses (bool) Plot a graph of losses during training
    :param: save_model (bool) Save model after performing the list methods
    :param: save_model_pth (str) Path where model is saved if save_model is True
    :param: verbose (int) Verbose
    """

    def __init__(
        self,
        steps,
        epochs=5,
        plot_losses=True,
        save_model=True,
        save_model_pth="./models/student.pt",
        verbose=0,
    ):
        self.steps = steps
        self.device = device
        self.verbose = verbose

        self.plot_losses = plot_losses
        self.save_model = save_model
        self.save_model_path = save_model_pth
        self._validate_steps()
        self.epochs = epochs

    def _validate_steps(self):
        name, process = zip(*self.steps)

        for t in process:
            if not hasattr(t, ("train_student", "prune", "quantize")):
                raise TypeError(
                    "All the steps must support at least one of "
                    "train_student, prune or quantize method, {} is not"
                    " supported yet".format(str(t))
                )

    def get_steps(self):
        return self.steps

    def _iter(self, num_steps=-1):
        _length = len(self.steps) if num_steps == -1 else num_steps

        for idx, (name, process) in enumerate(islice(self.steps, 0, _length)):
            yield idx, name, process

    def _fit(self):

        if self.verbose:
            pbar = tqdm(total=len(self))

        for idx, name, process in self._iter():
            print("Starting {}".format(name))
            if idx != 0:
                if hasattr(process, "train_student"):
                    if hasattr(self.steps[idx - 1], "train_student"):
                        process.student_model = self.steps[idx - 1].student_model
                    else:
                        process.student_model = self.steps[idx - 1].model
            t1 = time.time()
            if hasattr(process, "train_student"):
                process.train_student(
                    self.epochs, self.plot_losses, self.save_model, self.save_model_path
                )
            elif hasattr(proces, "prune"):
                process.prune()
            elif hasattr(process, "quantize"):
                process.quantize()
            else:
                raise TypeError(
                    "{} is not supported by the pipeline yet.".format(process)
                )

            t2 = time.time() - t1
            print(
                "{} completed in {}hr {}min {}s".format(
                    name, t2 // (60 * 60), t2 // 60, t2 % 60
                )
            )

            if self.verbose:
                pbar.update(1)

        if self.verbose:
            pbar.close()

    def train(self):
        """
        Train the (student) model sequentially through the list.
        """
        self._validate_steps()

        t1 = time.time()
        self._fit()
        t2 = time.time() - t1
        print(
            "Pipeline execution completed in {}hr {}min {}s".format(
                t2 // (60 * 60), t2 // 60, t2 % 60
            )
        )
