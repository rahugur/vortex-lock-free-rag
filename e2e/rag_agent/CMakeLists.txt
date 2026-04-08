# E2E: Forge Agent + Vortex Retrieval — single binary

add_executable(rag_agent main.cpp)

# Link against both Vortex (RAG engine) and Forge (agent runtime)
# forge-core is already linked transitively via vortex
target_link_libraries(rag_agent PRIVATE vortex)

# Include paths: Forge headers (for tools, llm, etc.) + Vortex headers
target_include_directories(rag_agent PRIVATE
    ${CMAKE_SOURCE_DIR}/forge/src     # Forge: tools/, llm/, core/
    ${CMAKE_SOURCE_DIR}/src           # Vortex: index/, pipeline/, etc.
    ${CMAKE_SOURCE_DIR}/e2e/rag_agent # knowledge_search_tool.h
)
