# llmparse
llmparse is an open-source Python script developed by DiscordDigital that generates questions from a given text input and runs them through a benchmark to evaluate the performance of an LLM (Language Learning Model). It can be used to obtain a question-and-answer based chat history for your LLMs, resulting in more precise responses.
## Installation
1. **Prerequisites**: You need to have Python 3.x installed on your system.
2. **Installation Steps**:
    - Clone this repository: `git clone https://github.com/DiscordDigital/llmparse`
    - Navigate into the cloned directory: `cd llmparse`
    - Install the requirements by running: `pip install -r requirements.txt`
    - Run the script: `python llmparse.py <input_file.txt>`
    - The script will prompt you, if you need to download any LLMs, adjust LLMs if needed in script.
## Usage
1. **Usage Example**: To run llmparse, navigate into the cloned directory and execute: `python llmparse.py <input_file>`.\
   Example, if you have a text file named "example.txt", use the following command: `python llmparse.py example.txt`
3. **Available Options**:
    - --to-history Converts an exported chat from OpenWeb UI to a python Ollama history file.\
      Example usage: `python llmparse.py <input_file.json> --to-history`
    - --to-modelfile Converts a file containing history data to a Modelfile (only messages).\
      Example usage: `python llmparse.py <input_file.json> --to-modelfile`
    - --question-generator Runs a text file of paragraphs through the question-generator.\
      Example usage: `python llmparse.py <input_file.txt> --question-generator`
    - --benchmark Runs a benchmark on a python Ollama history file.\
      Example usage: `python llmparse.py <input_file.json> --benchmark`

A file will be written to your current directory containing the output.
## Contributing
Contribute to llmparse by forking this repository and submitting pull requests with your changes.\
**Disclaimer**: Pull requests submitted to llmparse will be reviewed by DiscordDigital for potential acceptance. Contributions should be thoroughly tested before submitting to ensure quality.
## License
llmparse is an open-source project released under the MIT License.\
For more information about this license, please view the `LICENSE` file in this repository.
## Author
llmparse was developed by [DiscordDigital](https://github.com/DiscordDigital).\
If you have any questions, feedback, or issues regarding llmparse, please reach out to DiscordDigital on GitHub.
### Note
This README file was written by an LLM Modelfile generated by llmparse.
