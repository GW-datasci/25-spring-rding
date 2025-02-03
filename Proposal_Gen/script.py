import os
from datetime import datetime
import json
import shutil
import markdown2
import re
import mistune
# ....................................................................................................
input_file_path = 'input.json'
Year = "2024"
Semester = "Spring"
Version = "998"

class MyRenderer(mistune.Renderer):
    def list_item(self, text):
        # Wrap each list item in <li> tags and add a line break
        return f'<li>{text}</li>\n'

semester2code = { "sp":"01", "spr":"01", "spring":"01", "su":"02", "sum":"02", "summer":"02", "fa":"03", "fall":"03"}

def convert_md_to_html(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

        # Convert Markdown to HTML with mistune for better control
        renderer = MyRenderer()
        md_parser = mistune.Markdown(renderer)
        html_content = md_parser(md_content)

        # Keep spaces after numbers and dots in enumeration
        html_content = re.sub(r'(\d+)\. ', r'\1. ', html_content)

        with open(output_file, 'w', encoding='utf-8') as html_file:
            html_file.write(html_content)

def generate_readme(data, output_file_path):
    readme_template = \
f"""
# Capstone Proposal
## {data['project_name']}
### Proposed by: {data['Proposed by']}
#### Email: {data['Proposed by email']}
#### Advisor: {data['instructor']}
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  
{data['Objective']}

![Figure 1: Example figure]({Year}{semester2code[Semester.lower()]}_{data['Version']}.png)
*Figure 1: Caption*

## 2 Dataset:  
{data['Dataset']}

## 3 Rationale:  
{data['Rationale']}

## 4 Approach:  
{data['Approach']}

## 5 Timeline:  
{data['Timeline']}

## 6 Expected Number Students:  
{data['Expected Number Students']}

## 7 Possible Issues:  
{data['Possible Issues']}


## Contact
- Author: {data['instructor']}
- Email: [{data['instructor_email']}](Eamil)
- GitHub: [{data['github_repo']}](Git Hub rep)
"""
    # .......................................................
    with open(output_file_path + f"proposal_{data['Version']}.md", "w") as readme_file:
        readme_file.write(readme_template)
# end generate_readme()


# -------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # change these at the top of file
    # input_file_path = 'input.json'
    # Year = "2024"
    # Semester = "Spring"
    # Version = "999"

    output_file_path = os.getcwd() + f'{os.sep}Proposals{os.sep}{Year}{semester2code[Semester.lower()]}{os.sep}{Version}{os.sep}'
    with open(output_file_path+ input_file_path, 'r') as input_file:
        json_data = json.load(input_file)
    generate_readme(json_data, output_file_path)
    md_file = f"proposal_{json_data['Version']}.md"
    html_file = f"proposal_{json_data['Version']}.html"
    html_output = convert_md_to_html(output_file_path + md_file, output_file_path+ html_file)
    try:
        shutil.copy(f"{json_data['Year']}_{json_data['Semester']}_{json_data['Version']}.png",output_file_path )
    except:
        print('No Figure is needed.')
    print(f"proposal_{json_data['instructor']}.md generated successfully.")
