from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
from src.utils.evaluation_utils import evaluate_multiclass
import torch.nn as nn
import torch
import numpy as np
from src.optimizers.gd import GD
from torch import optim

criterion = nn.CrossEntropyLoss()
mseloss = nn.MSELoss()


class Worker(object):
    """
    Base worker for all algorithm. Only need to rewrite `self.local_train` method.

    All solution, parameter or grad are Tensor type.
    """

    def __init__(self, model, options):
        # Basic parameters
        self.model = model
        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        self.num_epoch = options['num_epoch']
        self.lr = options['lr']
        self.meta_lr = options['meta_lr']
        self.gpu = options['gpu'] if 'gpu' in options else False

        # Setup local model and evaluate its statics
        # self.flops, self.params_num, self.model_bytes = \
        #     get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])

    @property
    def model_bits(self):
        return self.model_bytes * 8

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y, _ in dataloader:
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x, y)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def local_train(self, train_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """

        self.model.train()

        y_total = []
        pred_total = []
        prob_total = []

        train_loss = 0
        for epoch in range(self.num_epoch):
            for batch_idx, (x, y, _) in enumerate(train_dataloader):
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                prob = self.model(x)

                loss = criterion(prob, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(prob, 1)
                train_loss += loss.item() * y.size(0)

                prob_total.append(prob.cpu().detach().numpy())
                pred_total.extend(predicted.cpu().numpy())
                y_total.extend(y.cpu().numpy())

        train_total = len(y_total)

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total}
        return_dict.update(param_dict)

        multiclass_eval_dict = evaluate_multiclass(y_total, pred_total, prob_total)
        return_dict.update(multiclass_eval_dict)

        return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = 0
        y_total = []
        pred_total = []
        prob_total = []
        with torch.no_grad():
            for x, y, _ in test_dataloader:
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                # prob = self.model(x)
                # loss = criterion(prob, y)
                prob = self.model(x, y)
                loss = criterion(prob, y)
                _, predicted = torch.max(prob, 1)

                prob_total.append(prob.cpu().detach().numpy())
                pred_total.extend(predicted.cpu().numpy())
                y_total.extend(y.cpu().numpy())

                test_loss += loss.item() * y.size(0)

        multiclass_eval_dict = evaluate_multiclass(y_total, pred_total, prob_total)
        return multiclass_eval_dict, test_loss


class LrdWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrdWorker, self).__init__(model, optimizer, options)

    def local_train(self, train_dataloader, **kwargs):
        # current_step = kwargs['T']
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch * 10):
            x, y, _ = next(iter(train_dataloader))

            if self.gpu:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict


class LrAdjustWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrAdjustWorker, self).__init__(model, optimizer, options)

    def local_train(self, train_dataloader, **kwargs):
        m = kwargs['multiplier']
        current_lr = self.optimizer.get_current_lr()
        self.optimizer.set_lr(current_lr * m)

        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch * 10):
            x, y, _ = next(iter(train_dataloader))

            if self.gpu:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            # lr = 100/(400+current_step+i)
            self.optimizer.step()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)

        self.optimizer.set_lr(current_lr)
        return local_solution, return_dict


class ImbaWorker(Worker):
    def __init__(self, model, options):
        self.meta_optimizer = optim.Adam(model.cs_parameters(), options['meta_lr'])
        super(ImbaWorker, self).__init__(model, options)

    def get_model_cs_params(self):
        return self.model.cs_parameters()[0].data.detach()

    def set_model_cs_params(self, cs_params):
        self.model.cs_parameters()[0].data.copy_(cs_params)

    def local_train(self, train_dataloader, val_dataloader, **kwargs):
        """Train model locally and return new parameter and computation cost

        Args:
            train_dataloader: DataLoader class in Pytorch
            val_dataloader: DataLoader class in Pytorch

        Returns
            1. local_solution: updated new parameter
            2. stat: Dict, contain stats
                2.1 comp: total FLOPS, computed by (# epoch) * (# data) * (# one-shot FLOPS)
                2.2 loss
        """

        self.model.train()

        y_total = []
        pred_total = []
        prob_total = []
        train_loss = 0

        # optimize on model parameters
        for epoch in range(self.num_epoch):
            for batch_idx, (x_train, y_train, train_idx) in enumerate(train_dataloader):
                # load data
                x_val, y_val, _ = next(iter(val_dataloader))
                if self.gpu:
                    x_train, y_train = x_train.cuda(), y_train.cuda()
                    x_val, y_val = x_val.cuda(), y_val.cuda()

                # _, tmp_loss = self.model(x_train, y_train, train_idx=train_idx)
                #
                # # temporarily update model parameter
                # grad = torch.autograd.grad(tmp_loss, self.model.parameters(), create_graph=True, retain_graph=True)
                # temp_new_model_weights = list(map(lambda p: p[1] - self.lr * p[0], zip(grad, self.model.parameters())))
                #
                # # update cost-sensitive model parameters with validation data
                # prob = self.model(x_val, y_val, vars=temp_new_model_weights, cs_weight=False)
                # loss = criterion(prob, y_val)
                #
                # # cs_grad = torch.autograd.grad(loss, self.model.cs_parameters())
                # # cs_model_weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(cs_grad, self.model.cs_parameters())))
                # # cs_model_weights[1] = torch.max(cs_model_weights[1], torch.tensor([0.]))
                # # cs_model_weights[1] = cs_model_weights[1] / torch.sum(cs_model_weights[1])
                #
                # self.meta_optimizer.zero_grad()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.cs_parameters(), 60)
                # self.meta_optimizer.step()
                # self.model.normalize_cs_parameters()

                # calculate loss
                prob, loss = self.model(x_train, y_train, train_idx=train_idx)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(prob, 1)
                train_loss += loss.item() * y_train.size(0)

                prob_total.append(prob.cpu().detach().numpy())
                pred_total.extend(predicted.cpu().numpy())
                y_total.extend(y_train.cpu().numpy())

        train_total = len(y_total)

        local_solution = self.get_flat_model_params()
        local_cs_solution = self.get_model_cs_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        # comp = self.num_epoch * train_total * self.flops
        return_dict = {#"comp": comp,
                       "loss": train_loss / train_total}
        return_dict.update(param_dict)

        multiclass_eval_dict = evaluate_multiclass(y_total, pred_total, prob_total)
        return_dict.update(multiclass_eval_dict)

        return local_solution, local_cs_solution, return_dict

