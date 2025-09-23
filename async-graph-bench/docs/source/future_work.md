# Future Work

* Concurrency using multiple threads for nodes so that they can be executed in parallel if resources are available
* Built in Error Handling (make it possible for nodes to mark items as errornous, which should result in these items being requeued into the nodes)
  * use case: LLMs may not respond in the desired format (for example providing 9 bullet point list items instead of expected 10). Recalculating within the node that receives batched items (where only 1 of 20 items may have an error is computationally expensive)
* Thorough testing of concurrency issues, specifically with the EndOfData signal (signal may arrive before all items are processed, currently not possible unless multiple resources are used, highly unlikely though.
* Implement more LLM inference APIs (currently OpenAI and vLLM are supported)

