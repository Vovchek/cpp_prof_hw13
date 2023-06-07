**build using docker on linux/WSL:**
- copy project folder
- start VS Code
- open project folder
- in VS Code terminal execute:
`# docker run --rm -ti -v $(pwd):/usr/src/app sdukshis/cppml`
`# rm -r build`
`# cmake -B build`
`# cmake --build build`
**build localy on windows/linuxs:**
`# cmake -B build`
`# cmake --build build`
**run fashio_mnist:**
`# ./build/fashio_mnist ./train/test.csv`
**run tests:**
`# ./build/test_fashio_mnist`
