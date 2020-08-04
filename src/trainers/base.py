import numpy as np
import torch
import time
from src.models.client import Client
from src.utils.worker_utils import Metrics
from src.models.worker import Worker, LrdWorker


class BaseTrainer(object):
    def __init__(self, options, dataset, model=None, optimizer=None, name='', worker=None):
        if model is not None and optimizer is not None:
            self.worker = Worker(model, optimizer, options)
        elif worker is not None:
            self.worker = worker
        else:
            raise ValueError("Unable to establish a worker! Check your input parameter!")
        print('>>> Activate a worker for training')

        self.gpu = options['gpu']
        self.batch_size = options['batch_size']
        self.lr_decay = options['lr_decay']
        self.all_train_data_num = 0
        self.clients = self.setup_clients(dataset)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.eval_every = options['eval_every']
        self.simple_average = not options['noaverage']
        print('>>> Weigh updates by {}'.format(
            'simple average' if self.simple_average else 'sample numbers'))

        # Initialize system metrics
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']
        self.latest_model_params = self.worker.get_flat_model_params()

    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Do not use gpu')

    def setup_clients(self, dataset):
        """Instantiates clients based on given train and test data directories

        Returns:
            all_clients: List of clients
        """
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
            all_clients.append(c)
        return all_clients

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def select_clients(self, seed=1):
        """Selects num_clients clients weighted by number of samples from possible_clients

        Args:
            1. seed: random seed
            2. num_clients: number of clients to select; default 20
                note that within function, num_clients is set to min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        """
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """

        self.worker.optimizer.soft_decay_learning_rate(self.lr_decay)

        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model_params)

            # Solve minimization locally
            soln, stat = c.local_train()
            # if self.print_result:
            #     print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
            #           "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
            #           "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
            #            round_i, c.cid, i, self.clients_per_round,
            #            stat['norm'], stat['min'], stat['max'],
            #            stat['loss'], stat['acc']*100, stat['time']))


            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def aggregate(self, solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        averaged_solution = torch.zeros_like(self.latest_model_params)
        # averaged_solution = np.zeros(self.latest_model.shape)
        # if self.simple_average:
        #     num = 0
        #     for num_sample, local_solution in solns:
        #         num += 1
        #         averaged_solution += local_solution
        #     averaged_solution /= num
        # else:
        num_sample_sum = 0
        for num_sample, local_solution in solns:
            averaged_solution += num_sample * local_solution
            num_sample_sum += num_sample
        averaged_solution /= num_sample_sum
        # averaged_solution *= (100 / self.clients_per_round)

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

    def test_latest_model_on_traindata(self, round_i):
        # Collect stats from total train data
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        # Record the global gradient
        model_len = len(self.latest_model_params)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        # Measure the gradient difference
        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)
        stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / MAvA: {:.3%} /'
                  ' eGmean: {:.3%} / MFM: {:.3%} / MAUC: {:.3%} / Loss: {:.4f} /'
                  ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(
                    round_i, stats_from_train_data['acc'], stats_from_train_data['mava'],
                    stats_from_train_data['egmean'], stats_from_train_data['mfm'],
                    stats_from_train_data['mauc'], stats_from_train_data['loss'],
                    stats_from_train_data['gradnorm'], difference, end_time - begin_time))
            print('=' * 102 + "\n")
        return global_grads

    def test_latest_model_on_evaldata(self, round_i):
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print('= Test = round: {} / Acc: {:.3%} / MAvA: {:.3%} /'
                  ' eGmean: {:.3%} / MFM: {:.3%} / MAUC: {:.3%} /'
                  ' Loss: {:.4f} / Time: {:.2f}s'.format(
                    round_i, stats_from_eval_data['acc'], stats_from_eval_data['mava'],
                    stats_from_eval_data['egmean'], stats_from_eval_data['mfm'],
                    stats_from_eval_data['mauc'], stats_from_eval_data['loss'],
                    end_time - begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

    def local_test(self, use_eval_data=True):
        assert self.latest_model_params is not None
        self.worker.set_flat_model_params(self.latest_model_params)

        num_samples = []
        test_eval_dict_list = []
        tot_test_eval_dict = {}
        losses = []
        for c in self.clients:
            test_eval_dict, num_sample, loss = c.local_test(use_eval_data=use_eval_data)

            test_eval_dict_list.append(test_eval_dict)
            num_samples.append(num_sample)
            losses.append(loss)

        ids = [c.cid for c in self.clients]
        groups = [c.group for c in self.clients]

        for key in test_eval_dict.keys():
            tot_test_eval_dict[key] = np.mean([x[key] for x in test_eval_dict_list])

        stats = {'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids, 'groups': groups}
        stats.update(tot_test_eval_dict)

        return stats
