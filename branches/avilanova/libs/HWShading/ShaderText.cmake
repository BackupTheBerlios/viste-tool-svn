MACRO(SHADER_TO_HEADER ShaderName ShaderFile HeaderFile)

FILE (READ ${ShaderFile} FILE_CONTENTS)

STRING(REGEX REPLACE
 "\n"
  "\\\\n"
  FILE_CONTENTS "${FILE_CONTENTS}"
)

STRING (REGEX REPLACE
  "\r"
  "\\\\r"
  FILE_CONTENTS "${FILE_CONTENTS}"
)

SET(OUTPUT_FILE ${HeaderFile})
SET(DEFINE_NAME ${HeaderFile})
STRING(REGEX REPLACE "/" "_" DEFINE_NAME ${DEFINE_NAME})
STRING(REGEX REPLACE "\\." "_" DEFINE_NAME ${DEFINE_NAME})
STRING(REGEX REPLACE ":" "_" DEFINE_NAME ${DEFINE_NAME})

FILE(WRITE ${OUTPUT_FILE} "#ifndef bmia_${DEFINE_NAME}\n")
FILE(APPEND ${OUTPUT_FILE} "#define bmia_${DEFINE_NAME}\n\n")

FILE(APPEND ${OUTPUT_FILE} "#define ${ShaderName} \"${FILE_CONTENTS}\"\n\n")

FILE(APPEND ${OUTPUT_FILE} "#endif // bmia_${DEFINE_NAME}\n")

ENDMACRO(SHADER_TO_HEADER)
