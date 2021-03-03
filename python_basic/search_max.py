#找最大最小

def search_max_min(a):
    max = a[0]
    min = a[0]
    for i in range(1,len(a)):
        if a[i] > max:
            max = a[i]
        if a[i] < min:
            min = a[i]
    return max,min

print(search_max_min([12,32,11,23,56,78,34]))