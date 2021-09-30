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

    # Vertice collections output-directory to export (file names will be the same as collection names)
    mkdir -p "$(pwd)/${databaseName}_export/vertices"
    folderVertices="$(pwd)/${databaseName}_export/vertices"
    
    # Edge collections output-directory to export (file names will be the same as collection names)
    mkdir -p "$(pwd)/${databaseName}_export/edges"
    folderEdges="$(pwd)/${databaseName}_export/edges"

	# Vertice collections output-directory to dump
    mkdir -p "$(pwd)/${databaseName}_dump"
    folderDump="$(pwd)/${databaseName}_dump"

	# Export documents from collections to json files
	# https://www.arangodb.com/docs/3.6/programs-arangoexport.html

	# Vertex collections 

		arangoexport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
        --output-directory "${folderVertices}" \
		--type json \
		--collection "datasources" \
		--collection "elements" \
		--collection "reactions" \
		--collection "reactionsets" \
		--collection "substances" \
		--collection "interactions" \
		--collection "mixingmodels" \
		--collection "phases" \
		--collection "thermodatasets" \
		--overwrite true	

	# Edge collections 
	
        arangoexport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
        --output-directory "${folderEdges}" \
		--type json \
        --collection "basis" \
		--collection "citing" \
		--collection "defines" \
		--collection "master" \
		--collection "prodreac" \
		--collection "product" \
		--collection "pulls" \
		--collection "specific" \
		--collection "mixing" \
		--collection "interacts" \
		--collection "component" \
        --collection "takes" \
		--overwrite true


	# Dump documents from collections to json files
	# https://www.arangodb.com/docs/3.6/programs-arangodump.html

	# Vertex collections 

		arangodump \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
        --output-directory "${folderDump}" \
		--overwrite true
		
#____________________________________[-]_ARANGODB_EXPORT_DATA
