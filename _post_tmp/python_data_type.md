#### float, int

* int, float, (unicode) -> 4 Byte
* list overhead -> 64 Byte
* int, float overhead -> 24 Byte
* 모든 primative도 객체로 인식해서 모두 20 Byte의 overhead가 붙음
  - np.float16: 26 Byte (24 + 2 Byte)
  - np.float32: 28 Byte (24 + 4 Byte)
  - np.float64: 32 Byte (24 + 8 Byte)

 * np.float32([5]).nbytes => 4 Byte
 * np.float32([5, 5]).nbytes => 8 Byte

 #### Byte

 * sys.getsizeof(b'') => 33 Byte
 * sys.getsizeof(b'5') => 34 Byte (원소가 1개 추가될 때마다 1 Byte씩 증가)

 #### Unicode

 * sys.getsizeof(u'') => 49 Byte
 * sys.getsizeof(u'5') => 58 Byte (원소가 1개 추가될 때 9 Byte 증가)
 * sys.getsizeof(u'56') => 51 Byte (원소가 1개 추가될 때마다 1 Byte씩 증가)
 * sys.getsizeof(u'123') => 52 Byte (원소가 1개 추가될 때마다 1 Byte씩 증가)
 * sys.getsizeof(u'가') => 76 Byte (27 Byte 증가)
 * sys.getsizeof(u'가나') => 78 Byte (원소가 1개 추가될 때마다 2 Byte씩 증가)
