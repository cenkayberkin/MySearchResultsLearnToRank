

a = ['url1' , 'url2', 'url3', 'url4', 'url5']

x = []
for i in range(0,len(a)):
  for b in range(i + 1,len(a)):
    x.append(a[i] +"-" + a[b])
    
print x


