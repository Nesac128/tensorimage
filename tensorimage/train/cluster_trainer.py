class ClusterTrainer:
    def __init__(self, **trainers):
        """
        :param trainers: Trainer object **kwargs with name (key) and object (value)
        """
        self.trainers = trainers

        self.trainer_data = {}
        for name in list(self.trainers.keys()):
            self.trainer_data[name] = {}
            self.trainer_data[name]["completed"] = False
        self.in_training = []

    def start(self):
        for (trainer, name) in zip(self.trainers.values(), self.trainers.keys()):
            self.train(trainer, name)

    def train(self, trainer, name):
        trainer.build_dataset()
        trainer.train()
        trainer.store_model()
        self.trainer_data[name]["name"] = name
        self.trainer_data[name]["testing_accuracy"] = trainer.final_testing_accuracy
        self.trainer_data[name]["testing_cost"] = trainer.final_testing_cost
        self.trainer_data[name]["n_epochs"] = trainer.n_epochs
        self.trainer_data[name]["learning_rate"] = trainer.learning_rate
        self.trainer_data[name]["l2_regularization_beta"] = trainer.l2_beta
        self.trainer_data[name]["train_test_split"] = trainer.train_test_split
        self.trainer_data[name]["architecture"] = trainer.architecture
        self.trainer_data[name]["batch_size"] = trainer.batch_size
        self.trainer_data[name]["completed"] = True

    def get_results(self):
        trainer_performance = {}
        accuracy_history = []

        for td in list(self.trainer_data.values()):
            accuracy_history.append((td["name"], td["testing_accuracy"]))
        accuracy_history.sort(key=lambda top: top[1], reverse=True)
        for n, data in enumerate(accuracy_history):
            trainer_performance[str(n+1)] = self.trainer_data[data[0]]

        return trainer_performance
