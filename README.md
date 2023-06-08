**build using docker on linux/WSL:**
- copy project folder
- start VS Code
- open project folder
- in VS Code terminal execute:
```
# docker run --rm -ti -v $(pwd):/usr/src/app sdukshis/cppml
# rm -r build
# cmake -B build
# cmake --build build
```

**build localy on windows/linuxs:**
```
# cmake -B build
# cmake --build build
```
**run fashio_mnist:**
```
# ./build/fashio_mnist ./train/test.csv
```
***note:** program needs to load matrixces of sizes\
 784x128 && 128x10 from **w1.txt** && **w2.txt** respectively,\
 located at **./train** folder.*
 
**run tests:**
```
# ./build/test_fashio_mnist
```
