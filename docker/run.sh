docker run \
	-it \
	-v "$(pwd)":/lab \
	-p 8888:8888 \
	--rm \
	--user "$(id -u)" \
	--group-add users \
	odp
