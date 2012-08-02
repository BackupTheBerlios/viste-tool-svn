#!/bin/bash

find -L . -type f -name "*.cxx" | while read CXX; do
  mv "$CXX" "$CXX.backup"
  cat license_cxx.txt $CXX.backup > $CXX
  rm $CXX.backup
done

find -L . -type f -name "*.h" | while read H; do
  mv "$H" "$H.backup"
  cat license_cxx.txt $H.backup > $H
  rm $H.backup
done

find -L . -type f -name "CMakeLists.txt" | while read TXT; do
  mv "$TXT" "$TXT.backup"
  cat license_cmake.txt $TXT.backup > $TXT
  rm $TXT.backup
done
