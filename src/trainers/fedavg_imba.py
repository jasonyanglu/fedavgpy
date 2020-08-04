from src.trainers.base import BaseTrainer
from src.models.model import choose_model, CostSensitiveWeight
from src.models.worker import ImbaWorker
from src.optimizers.gd import GD


class FedAvgTrainerImba(BaseTrainer):
    def __init__(self, options, dataset):
        model = choose_model(options)
        self.cs_model = CostSensitiveWeight(options['num_class'] ** 2)
        self.move_model_to_gpu(model, options)
        self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        worker = ImbaWorker(model, self.optimizer, options)
        super(FedAvgTrainerImba, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)
            self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        self.test_latest_model_on_traindata(self.num_round)
        self.test_latest_model_on_evaldata(self.num_round)

        # Save tracked information
        self.metrics.write()


