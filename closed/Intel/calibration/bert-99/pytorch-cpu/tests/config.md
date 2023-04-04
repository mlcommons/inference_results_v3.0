user=$USERNAME

mount -t hugetlbfs -o uid=$(id -u $user),gid=$(id -g $user),mode=0700,pagesize=2m,size=100g none ./huge
