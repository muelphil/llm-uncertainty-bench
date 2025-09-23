# Sampling

Sampling refers to grouping of several iterations of items together, as specified by the `sampling_config` in the `NodeConfig`, so that statistics based on samples of data for the same item may be calculated. This is relevant if nodes are used that feature randomness (such as noise or LLM inference).

Manager specifies total iterations per item provided by data_source. Sampling layer (see [Node Configuration](node_configuration)) is responsible for grouping items with same id together.

`sampling_config` may be provided in the NodeConfig:
* `sampling_size` specifies the amount of items to be sampled. `sampling_size` must be a divisor of the total amout of iterations. If iter is 10 and sampling_size is 5, the 10 items will be sampled into 2 batches of 5 items each.
* `all_variations` specifies the sampling mode, if it is not `spread` (see below). If `all_variations` is True, sampling mode will be "extend", else "first only"
sampling is activated if Nodes use a sampling dependency, as indicated by the prefix `sampling_`. Sampling will be done for these specific dependencies. They will be provided to the nodes as dependencies that are a list with an element for each provided (sampled) item, where the elements are yet again lists holding the sampled dependencies of the items within the sampled batch. 

## Sampling Modes
There are 3 modes for sampling.

### "First only" sampling (default)
In this mode, only the lowest iteration of each sampling batch will be extended by the sampled dependencies and provided to the node. For example, for 10 iterations and `sampling_size` 5, the items of iteration 0 and 5 will be extended by the sampled_dependency. This is destructive, as the node will only emit the items of iteration 0 and 5 as the result of this node. Nodes depending on a node that is sampling in first only mode will therefore only receive a subset of items. If subsequent nodes also want to use sampling, they need to use sampling in "First only" mode and can only sample dependencies that are available to the first "First only" sampling node in the chain. Sampling will then exclusively be performed by the first "First only" sampling node, sampling dependencies for itself and subsequent "First only" nodes.

Example:
```python
import statistics
class DiffFromMean:
    """
    Calculates how far each item's score is from the mean score of its sampled batch.

    Dependencies:
        - score: A list of individual item scores.
        - sampled_score: A list of lists, where each sublist contains the scores of 
          items in the corresponding sample batch.

    Output:
        - diff_from_mean: A list of absolute differences between each item's score 
          and the mean of its corresponding sampled_score batch.
    """
    dependencies = ["score", "sampled_score"]

    async def __call__(self, dependencies: Dict[str, list]) -> Dict[str, List]:
        return {
            "diff_from_mean": [
                abs(val - statistics.mean(sample))
                for val, sample
                in zip(dependencies["score"], dependencies["sampled_score"])
            ]
        }
```
usage:
```python
from async_graph_bench import NodeConfig, SamplingConfig
NodeConfig(
    DiffFromMean(),
    sampling_config=SamplingConfig(sampling_size=3, all_variations=False)
)
```

For items
```python
{"id": 0, "iter": 0, "score": 5},
{"id": 0, "iter": 1, "score": 3},
{"id": 0, "iter": 2, "score": 9},
```
 and `sample_size=3` (and batching according to `NodeConfig` disabled), the node would be called once receiving the dependencies
```python
{"score": [5], "sampled_score": [[5,3,9]]} # id=0, iter=0
```



### "Extend" sampling
This mode is similar to "first only" sampling but will extend all items of a sampling batch with the sampled dependencies. All items of a sampling batch are then provided to node for calculation. This is non destructive. The sampled dependencies (represented by lists holding the corresponding dependency values for each item in the sampling batch) will be shifted for each iteration, so that the dependency of the item that is extended is always first in the list. 

This does not require any changes in the implementation of the node. The same implementation may be used, simply specifying in the config:
usage:
```python
from async_graph_bench import NodeConfig, SamplingConfig
NodeConfig(
    DiffFromMean(),
    sampling_config=SamplingConfig(sampling_size=3,
                                   all_variations=True) # <-- True
)
```

For items
```python
{"id": 0, "iter": 0, "score": 5},
{"id": 0, "iter": 1, "score": 3},
{"id": 0, "iter": 2, "score": 9},
```
 and `sample_size=3` (and batching disabled according to `NodeConfig`), the node would be called 3 times with the dependencies
```python
{"score": [5], "sampled_score": [[5,3,9]]} # id=0, iter=0
{"score": [3], "sampled_score": [[3,9,5]]} # id=0, iter=1
{"score": [9], "sampled_score": [[9,5,3]]} # id=0, iter=2
```

### "Spread" sampling
"Extend" sampling is nice to calculate sampling metrics that are dependent on the individual items. However, it often is possible to easily calculate the sampling metric for all items of a sampling batch at once, instead of individually for the items.
"Spread" sampling makes this possible. Spread sampling is activated by the node specifying the `spread` attribute as `True`. Similar to "First only" sampling, the node will only receive the first item in the sampling batch with the required sampled dependencies. It may then calculate the scores for all items in the sampled batch (although only provided once, in contrast to "Extend" sampling) and the results will be put/ set in the individual iterations of the item afterwards. This makes sense if the resulting scores are only dependend on the sampled dependency of the items, but not of other dependencies.
As a return type for the individual stats provided by the node, the node must then for each item (representing the sampled_batch) processed return as values of the result (keys are dependencies) lists of the length of the sample size with the results of the calculation for each iteration based on the sample batch provided. These results will then be automatically be spread by the sampling layer. This makes calculation of metrics using sampling that can easily be calculated in batch easier.

Example:
```python
import statistics
class DiffFromMeanSpread:
    """
    Computes, for each sample batch, how far every score in that batch deviates 
    from the batch’s mean score.

    Dependencies:
        - sampled_score: A list of lists, where each sublist contains the scores 
          of items in one sample batch.

    Properties:
        - spread = True: Indicates that the output is structured as lists of values 
          for each batch (rather than one value per item).

    Output:
        - diff_from_mean: A list of lists, where each inner list contains the 
          absolute differences between every score in the sample and the mean 
          of that sample.
    """
    dependencies = ["sampled_score"]
    spread = True

    async def __call__(self, dependencies: Dict[str, list]) -> Dict[str, List]:
        return {
            "diff_from_mean": [
                [abs(score - statistics.mean(sample)) for score in sample] # <-- returning list of scores, one for each item in the sample batch
                for sample
                in dependencies["sampled_score"]
            ]
        }
```