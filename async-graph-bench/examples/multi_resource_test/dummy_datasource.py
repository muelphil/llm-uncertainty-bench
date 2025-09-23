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


class DummyDataSource(DataSource):
    def __init__(self):
        super().__init__(["input_texts"])

    def __len__(self):
        return len(questions)

    async def iter_items(self):
        for idx, question in enumerate(questions):
            yield {
                "id": idx,
                "input_texts": question
            }

    def iter_keys(self):
        for idx in range(len(questions)):
            yield idx
