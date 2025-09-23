# LLM Models
* this framework comes with an abstraction of LLM inference APIs
* base class `Model` and expected response type `ResponseWrapper` define interface 
* this was implemented for 2 APIs:
  * `VLLMModel` - inference using vLLM
  * `OpenAIAPIModel` - inference by querying an  openai api endpoint
* allows easy interchangeability of models in benchmark according requirements and hardware availability
* more models for different interfaces may be implemented in the future
* 
