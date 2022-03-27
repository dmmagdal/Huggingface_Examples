# preprocess.py
# Preprocess sample text data (stories) from literotica_crawler. This
# preprocessing is limited to removing prepended labels in the text
# files (such as "Description: ", "Story Link: ", "Tags: ", etc).
# Windows/MacOS/Linux
# Python 3.7


import os


def main():
	# The dataset consists of around 100 stories pulled from the
	# website with the literotica_crawler module and stored in a folder
	# (labeled "stories" in this case).
	folder = "./stories"
	output_folder = "./cleaned_stories" # Ironic given the content.
	os.makedirs(output_folder, exist_ok=True)
	for file in os.listdir(folder):
		with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
			contents = f.readlines()

			# Prepend special character to title.
			contents[0] = "<|startoftext|>" + "<|title|>" + contents[0].rstrip("\n")

			# indices 1, 2, 3, 4 are the author, description, story
			# link and author page link respectively. We do not want
			# those for the dataset. This is format is constant from
			# the way literotica_crawler saves stories.

			# Prepend special character to tags.
			contents[5] = "<|tags|>" + contents[5].rstrip("\n")

			# Prepend special character to main story.
			contents[6] = "<|story|>" + " ".join(
				[content.rstrip("\n") for content in contents[6:]]
			) + "<|endoftext|>"

		# Write the cleaned text to the output folder.
		contents = [contents[0]] + [contents[5]] + contents[6:]
		with open(os.path.join(output_folder, file), "w+", encoding="utf8") as f_out:
			f_out.write(" ".join(contents))

	# Exit the program.
	exit(0)


if __name__ == "__main__":
	main()