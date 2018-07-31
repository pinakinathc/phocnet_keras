'''This code will take an input word as in string and will
output the PHOC label of the word. The Phoc label is a
vector of length 604.
Reference: https://ieeexplore.ieee.org/document/6857995/?part=1

Author: Pinaki Nath Chowdhury <pinakinathc@gmail.com>
'''

def generate_36(word):
	'''The vector is a binary and stands for:
	[0123456789abcdefghijklmnopqrstuvwxyz]
	'''
	vector_36 = [0 for i in range(36)]
	for char in word:
		if char.isdigit():
			vector_36[ord(char) - ord('0')] = 1
		elif char.isalpha():
			vector_36[10+ord(char) - ord('a')] = 1

	return vector_36

def generate_50(word):
	'''This vector is going to count the number of most frequent
	bigram words found in the text:
	['th', 'he', 'in', 'er', 'an', 're', 'es', 'on', 'st', 'nt', 'en',
	'at', 'ed', 'nd', 'to', 'or', 'ea', 'ti', 'ar', 'te', 'ng', 'al',
	'it', 'as', 'is', 'ha', 'et', 'se', 'ou', 'of', 'le', 'sa', 've',
	'ro', 'ra', 'hi', 'ne', 'me', 'de', 'co', 'ta', 'ec', 'si', 'll',
	'so', 'na', 'li', 'la', 'el', 'ma']

	Reference: http://practicalcryptography.com/media/cryptanalysis/files/english_bigrams_1.txt
	'''

	bigram = ['th', 'he', 'in', 'er', 'an', 're', 'es', 'on', 'st', 'nt', 'en',
  'at', 'ed', 'nd', 'to', 'or', 'ea', 'ti', 'ar', 'te', 'ng', 'al',
  'it', 'as', 'is', 'ha', 'et', 'se', 'ou', 'of', 'le', 'sa', 've',
  'ro', 'ra', 'hi', 'ne', 'me', 'de', 'co', 'ta', 'ec', 'si', 'll',
  'so', 'na', 'li', 'la', 'el', 'ma']

	vector_50 = [0 for i in range(50)]
	for char in word:
		try:
			vector_50[bigram.index(char)] = 1
		except:
			continue

	return vector_50

def generate_label(word):
	word = word.lower()
	vector = []
	L = len(word)
	for split in range(2, 6):
		parts = L//split
		for mul in range(split-1):
			vector += generate_36(word[mul*parts:mul*parts+parts])
		vector += generate_36(word[(split-1)*parts:L])

	# Append the most common 50 bigram text using L2 split
	vector += generate_50(word[0:L//2])
	vector += generate_50(word[L//2: L])
	return vector
