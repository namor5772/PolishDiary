import os
import re

def main():
    # Base directory is where this script resides
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Subdirectory containing transcripts
    transcripts_dir = os.path.join(base_dir, "output_transcripts")

    # Output file
    output_file = os.path.join(base_dir, "Eugenia_all.txt")

    # Regex to match files like Eugenia_pg###.txt
    pattern = re.compile(r"Eugenia_pg(\d+)\.txt$")

    # Collect all matching files
    files = []
    for f in os.listdir(transcripts_dir):
        match = pattern.match(f)
        if match:
            page_num = int(match.group(1))
            files.append((page_num, os.path.join(transcripts_dir, f)))

    # Sort files by the numeric part
    files.sort(key=lambda x: x[0])

    # Concatenate contents with page headers
    with open(output_file, "w", encoding="utf-8") as outfile:
        for page_num, filepath in files:
            header = f"\n===== Page {page_num:03d} =====\n"
            outfile.write(header)
            with open(filepath, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())
                outfile.write("\n")  # Ensure spacing after page text

    print(f"Combined {len(files)} files into {output_file}")

if __name__ == "__main__":
    main()
