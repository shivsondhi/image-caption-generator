def append_file_fn(read_file, append_file):
	file_r = open(read_file, 'rb')
	data = file_r.read()
	file_r.close()

	file_a = open(append_file, 'ab')
	file_a.write(data)
	file_a.close()

	return("Complete")

if __name__ == "__main__":
	for i in range(1,14):
		read_file = 'features\\features' + str(i) + '.pkl'
		append_file = 'all_features.pkl'
		print(append_file_fn(read_file, append_file))

	print("Done!")

	'''file_p = open(append_file, 'rb')
	print(file_p.read())
	file_p.close()'''