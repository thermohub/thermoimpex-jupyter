#!/bin/bash

#____________________________________[+]_ARANGODB_EXPORT_DATA

# ArangoDB Shell Configuration
# https://www.arangodb.com/docs/3.6/programs.html

	# Access credentials
	serverEndpoint=tcp://127.0.0.1:8529
	userName=root
	userPassword=""
	# Export data from database given as fisrt argument when executing the script
    databaseName="$1"

	# Vertice collections output-directory to dump
    folderDump="$(pwd)/hub_main_dump"

	# Vertex collections 

		arangorestore \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
                --create-database true \
                --input-directory "${folderDump}"
		
#____________________________________[-]_ARANGODB_EXPORT_DATA
