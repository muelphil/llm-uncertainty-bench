from async_graph_bench import DataSource

questions = [
    "Please list all the states of the United States of America and provide detailed information about each state's history, geography, culture, economy, demographics, notable cities, famous people, major industries, natural landmarks, and unique traditions in a narrative format.",
    "Describe the history of human civilization in as much detail as possible, starting from the earliest known records to the modern era, including major events, inventions, cultural movements, and influential figures.",
    "Write a comprehensive encyclopedia entry for every planet, moon, and celestial object in the solar system, including detailed descriptions of their physical characteristics, exploration history, and any associated mythology or cultural significance.",
    "For every known species of animal, provide an exhaustive overview, including their taxonomy, physical description, habitat, behavior, diet, reproduction, role in ecosystems, conservation status, and their interactions with humans.",
    "List every notable invention or scientific discovery in history and provide a detailed account of its inventor(s), the process of its development, its impact on society, and how it has evolved over time.",
    "Create an extensive travel guide for the entire world, including every country, major city, and famous landmarks, as well as recommendations for food, activities, cultural etiquette, and hidden gems in each location.",
    "Provide a detailed literary analysis of every major work of Shakespeare, including summaries, character breakdowns, themes, historical context, and interpretations of its lasting influence on modern literature and culture.",
    "Explain the entire history and development of computing and technology, covering every milestone from ancient computational tools to the modern era of quantum computing and artificial intelligence, including the people, companies, and societal impacts involved.",
    "For each country in the world, provide an in-depth account of its history, political system, geography, economy, demographics, culture, traditions, major challenges, and contributions to the global community.",
    "Explain the complete biology and ecology of coral reefs, detailing every species that relies on this ecosystem, the processes that sustain it, the threats it faces, and the efforts being made to preserve it."
]


class PromptDataSource(DataSource):
    """
    A minimal example of a data source that provides a fixed set of extremely large and complex input prompts.

    This source is intended for stress-testing or benchmarking LLM behavior on
    large, open-ended generation tasks. Each prompt requests the model to
    produce long, detailed responses on various expansive topics (e.g., world
    history, species overview, or planetary encyclopedia entries).
    """

    description = "Provides 10 extremely long prompts designed to test LLM performance and stability on large, open-ended generation tasks."
    provides = ["input_texts"]

    def __len__(self):
        """Return the total number of available questions."""
        return len(questions)

    async def iter_items(self):
        """
        Asynchronously iterate over all prompts and yield them as data items.

        Each yielded item includes:
          - 'id': a stable hash of the prompt string
          - 'input_texts': the actual question text to be passed to the model
        """
        for question in questions:
            yield {
                "id": hash(question),
                "input_texts": question,
            }

    def iter_ids(self):
        """
        Iterate over all data item IDs.

        Useful for identifying existing results in data stores
        without loading full prompt contents.
        """
        for question in questions:
            yield hash(question)
