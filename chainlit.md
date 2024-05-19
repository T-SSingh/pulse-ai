## Pulse AI

![PulseAI](https://raw.githubusercontent.com/arthikrangan/DocRegistry/main/content_markerting.png "Content Markerting")

This project is designed to generate personalized digital marketing content for B2B channels such as emails and social platforms.

Here are a few sample prompts to try - 
* Research Amgen Inc's sponsorship deals from the web and draft an outreach email requesting their sponsorship for a Rare Disease Summit. Highlight event benefits.

* Draft a welcome email for an event attendee, Mr. Jacob Harrison, who is interested in wound care, including a list of exhibitors specializing in wound care.

### Setup
1. Clone this repository.
2. Install the required Python packages: 
`pip install -r requirements.txt`
3. Create a `.env` file in the project root directory with the necessary environmental variables:
- `TAVILY_API_KEY`
- `OPENAI_API_KEY`
- `QDRANT_CLOUD_URL`
- `QDRANT_API_KEY`

NOTE: The qdrant cloud URL refers to a database that has Open AI embeddings of event exhibitor interests and specialities from the dataset - dataset/exhibitor_profiles.csv.

### Running the Application
Run the application by executing:
`chainlit run app.py --port 7680`

### Requirements
Python 3.11 or newer. 

### Support and Contact
If you need support with the project or have any queries, feel free to reach out to me.
[Arthi Kasturirangan](https://www.linkedin.com/in/arthikrangan/)