## Instruction to Generate Proposal

### 0. Make a copy of json_gen_ver.py 
- Replace _ver with a unique 3-digit serial version number (as string). Could be from "000" to "999".
### 1. Add required information into json_gen_xxx.py by modifying required field
- Year
- Semester
- Version
- project_name
- Objective
- Dataset
- Rationale
- Approach
- Expected Number Students
- Proposed by
- Proposed by email
- instructor
- instructor_email
- github_repo
### 2. Run the  json_gen_ver.py, 
- This will generate input.json in the appropriate folder.
### 3. Optional - Add image file
- If you have image to add, put it directly to the appropriate folder now, and name the src as yyyyss_ver.png
- If you are not using png, either convert it into png (preferred), or hard-code it accordingly.
- The universal semester code "ss" used is { "spring":"01", "summer":"02", "fall":"03" }.
- For example, the image would be named 202401_999.png
### 3. Run the  script.py then you should have the markdown file.
- Modify Year, Semester and Version Accordingly.
- If you need to have more pictures please modify the script.py directly.
