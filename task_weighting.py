import torch
import yaml
import argparse


def _multiples_and_weights(task_weight_exponent,dataset_sizes):
    """
    Helper for weighting GLUE datasets in PyTorch.

    Concatenating all the train sets together and then shuffling the examples
    causes large datasets to dominate the training, resulting in poor performance
    on small datasets. This has some logic to produce (1) "multiples" for
    each dataset so the multi-task train set contains small datasets multiple
    times, so those examples are seen more often and (2) weights for each dataset,
    which also allows for smaller datasets to have influence on training.
    Overall the effect is that tasks are weighted according to
    dataset_size^config.task_weight_exponent.

    Args:
        config: a configuration object with `task_weight_exponent` attribute
    Returns:
        How many copies and weights for each dataset.
    """

    # Dataset sizes
    # dataset_sizes = {
    #     "0": 4256,
    #     "1": 1686,
    #     "2": 652,
    #     "3": 186,
    #     "4": 889,
    #     "5": 176,
    #     "6": 323,
    #     "7": 3857,
    #     "8": 166,
    #     "9": 67,
    #     "10": 209,
    #     "11": 761,
    #     "12": 253
    #
    # }

    # Helper functions
    def normalize(d):
        total = torch.sum(torch.tensor(list(d.values()), dtype=torch.float))
        return {k: v / total for k, v in d.items()}

    # Calculate dataset weights
    dataset_weights = {k: v ** task_weight_exponent for k, v in dataset_sizes.items()}
    dataset_weights = normalize(dataset_weights)

    # Apply correction to keep MNLI in proportion
    correction = dataset_sizes[0] / dataset_weights[0]
    dataset_tgts = {k: v * correction for k, v in dataset_weights.items()}

    # Calculate dataset multiples
    dataset_multiples = {
        task: int(round((tgt.item() + 0.01) / dataset_sizes[task])) for task, tgt in dataset_tgts.items()
    }
    new_dataset_sizes = {task: dataset_sizes[task] * multiple for task, multiple in dataset_multiples.items()}

    # Normalize weights again after applying multiples
    weights_after_multiples = normalize({
        task: dataset_weights[task] / new_dataset_sizes[task] for task in new_dataset_sizes
    })
    weights_after_multiples = {k: v * len(dataset_sizes) for k, v in weights_after_multiples.items()}

    return dataset_multiples, weights_after_multiples


def get_task_multiple(task, split):
    if split != "train":
        return 1
    if task.config.dataset_multiples:
        multiples, _ = _multiples_and_weights(task.config)
        return int(multiples[task.name] + 1e-5)
    return 1


def get_task_weights(configs,sizes):
    """
    Get task weights according to dataset sizes.
    """

    if configs.bam.dataset_multiples:
        multiples, weights = _multiples_and_weights(configs.bam.task_weight_exponent,sizes)
        return multiples,weights
    else:
        multiples={}
        if configs.bam.task_weight_exponent < 0:
            return multiples,{task_name: 1.0 for task_name in sizes}

        n_examples = sum(sizes.values())
        weights = {task_name: 1.0 / (size ** (1 - configs.bam.task_weight_exponent)) for task_name, size in sizes.items()}
        expected_weight = sum([weights[task_name] * sizes[task_name] / n_examples for task_name in weights])
        weights = {task_name: w / expected_weight for task_name, w in weights.items()}
        return multiples,weights

if __name__ == '__main__':

    dataset_multiples=False
    task_weight_exponent=0.75

    sizes={"rte":100,"mrpc":120}

    weights= get_task_weights(dataset_multiples,task_weight_exponent,sizes)
    print(weights)