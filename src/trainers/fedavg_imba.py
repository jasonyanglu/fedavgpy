from src.trainers.base import BaseTrainer
import torch
import numpy as np
import copy


class FedAvgTrainerImba(BaseTrainer):
    def __init__(self, options, dataset):
        super(FedAvgTrainerImba, self).__init__(options, dataset)
        self.latest_model_cs_params = self.clients[0].worker.get_model_cs_params()

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        # self.latest_model_params = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)

            # Solve minimization locally
            solns, cs_solns, stats = self.local_train(round_i, selected_clients)

            # Track communication cost
            # self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model_params = self.aggregate(solns)
            self.latest_model_cs_params = self.aggregate_cs(cs_solns)
            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()

    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """

        solns = []  # Buffer for receiving client solutions
        cs_solns = []
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.worker.optimizer.soft_decay_learning_rate(self.lr_decay)
            c.worker.set_flat_model_params(self.latest_model_params)
            c.worker.set_model_cs_params(self.latest_model_cs_params)

            # Solve minimization locally
            soln, cs_soln, stat = c.local_train()

            # Add solutions and stats
            solns.append(soln)
            cs_solns.append(cs_soln)
            stats.append(stat)

        return solns, cs_solns, stats

    def local_test(self, use_eval_data=True):
        # assert self.latest_model_params is not None
        # self.worker.set_flat_model_params(self.latest_model_params)

        num_samples = []
        test_eval_dict_list = []
        tot_test_eval_dict = {}
        losses = []
        for c in self.clients:
            c.worker.set_flat_model_params(self.latest_model_params)
            c.worker.set_model_cs_params(self.latest_model_cs_params)
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

    def aggregate_cs(self, cs_solns, **kwargs):
        """Aggregate local solutions and output new global parameter

        Args:
            cs_solns: a generator or (list) with element (num_sample, local_solution)

        Returns:
            flat global model parameter
        """

        # averaged_solution = torch.zeros_like(self.latest_model_cs_params)
        # return_solution = copy.deepcopy(self.latest_model_cs_params)
        # class_num = self.latest_model_cs_params.shape[0]
        # label_count = torch.zeros(class_num)
        #
        # for labels, local_solution in cs_solns:
        #     unique_label = np.unique(labels)
        #     for i in unique_label:
        #         label_i_sum = sum(i == labels)
        #         averaged_solution[i] += label_i_sum * local_solution[i]
        #         label_count[i] += label_i_sum
        #
        # for i in range(class_num):
        #     if label_count[i] != 0:
        #         return_solution[i] = averaged_solution[i] / label_count[i]
        #         return_solution[i] = return_solution[i] / sum(return_solution[i]) * class_num

        return_solution = torch.ones_like(self.latest_model_cs_params)

        return return_solution.detach()

