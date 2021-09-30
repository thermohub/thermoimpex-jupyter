#!/bin/bash

#____________________________________[+]_ARANGODB_IMPORT_DATA

# ArangoDB Shell Configuration
# https://www.arangodb.com/docs/3.6/programs.html


	# Access credentials
	serverEndpoint=tcp://127.0.0.1:8529
	userName=root
	userPassword=""
	# (re)import data from json files into 'hub_main' database 
	#   (if database name is different, edit it in the arangosh script below)
        databaseName="$1"
        folderName="$2"

    # Vertice collections files to upload
    datasources="$(pwd)/${folderName}/vertices/datasources.json"
    elements="$(pwd)/${folderName}/vertices/elements.json"
    reactions="$(pwd)/${folderName}/vertices/reactions.json"
    reactionsets="$(pwd)/${folderName}/vertices/reactionsets.json"
    substances="$(pwd)/${folderName}/vertices/substances.json"
    thermodatasets="$(pwd)/${folderName}/vertices/thermodatasets.json"
    interactions="$(pwd)/${folderName}/vertices/interactions.json"
    mixingmodels="$(pwd)/${folderName}/vertices/mixingmodels.json"
    phases="$(pwd)/${folderName}/vertices/phases.json"
    
    # Edge collections files to upload
    basis="$(pwd)/${folderName}/edges/basis.json"
    citing="$(pwd)/${folderName}/edges/citing.json"
    defines="$(pwd)/${folderName}/edges/defines.json"
    master="$(pwd)/${folderName}/edges/master.json"
    prodreac="$(pwd)/${folderName}/edges/prodreac.json"
    product="$(pwd)/${folderName}/edges/product.json"
    pulls="$(pwd)/${folderName}/edges/pulls.json"
    takes="$(pwd)/${folderName}/edges/takes.json"
    specific="$(pwd)/${folderName}/edges/specific.json"
    mixing="$(pwd)/${folderName}/edges/mixing.json"
    interacts="$(pwd)/${folderName}/edges/interacts.json"
    component="$(pwd)/${folderName}/edges/component.json"

	# Create arangosh executable within USER home directory
	tee ~/.arangosh.import_data > /dev/null <<- 'END_.arangosh.import_data'
	#!/usr/bin/arangosh --javascript.execute

	//____________________________________[+]_IMPORT_DATA

	require('internal').print('hello world')
	// Imports
	const users = require('@arangodb/users')

	// Create databases
	// try {
	// db._createDatabase('hub_main')
	// } catch (err) {}

	// Create database users
    // const sysExtra = { 'name': 'ThermoSys user' }
	// users.save('sysloc', '', true, sysExtra)

	// Grant database to users
	// Root default access for all new databases
	// users.grantDatabase('root', '*', 'rw');

	// Root-like access, i.e. also to '_system' DB
	// users.grantDatabase('adminloc', '_system', 'rw');
	// users.grantDatabase('adminloc', '*', 'rw');

	// Permissive-first policy for database management clients
    // users.grantDatabase('sysloc', '_system', 'none');
	// users.grantDatabase('sysloc', '*', 'rw');

	// Restrictive-first policy for data delivery clients
	// users.grantDatabase('funloc', '_system', 'none');
	// users.grantDatabase('funloc', '*', 'none');
	// users.grantDatabase('funloc', 'hub_main', 'ro');
    // users.grantDatabase('funloc', 'hub_work', 'rw');

    // TBD more access rights for users to databases

	// Create collections using DB object API:
	// https://docs.arangodb.com/3.4/Manual/Programs/Arangosh/Options.html

	// Select DB instance
	db._useDatabase('hub_main')

	// Create collections for vertices
	// db._create(<name>, <properties>)
    // db._create('datasources')

	// Create collections for edges
    // db._createEdgeCollection(<name>, <properties>)
    // db._createEdgeCollection('citing')

	// Grant parmissions on collections to users
    // users.grantCollection('funloc', 'hub_main', 'datasources', 'ro')
    // users.grantCollection('funloc', 'hub_main', 'citing', 'ro')

    // TBD more granting to add, if needed

	//____________________________________[-]_SEED_DATA
	END_.arangosh.import_data

	# Make the file executable
	chmod a+x ~/.arangosh.import_data

	# Execute file with DB arguments
	~/.arangosh.import_data \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}"

	# Import documents into collections
	# https://www.arangodb.com/docs/3.6/programs-arangodump.html

	# Vertex collections 

		# 'datasources':
	 	arangoimport \
	 	--server.endpoint "${serverEndpoint}" \
	 	--server.username "${userName}" \
	 	--server.password "${userPassword}" \
	 	--server.database "${databaseName}" \
	 	--collection "datasources" \
     	--on-duplicate "replace" \
	 	--file "${datasources}" \
     	--batch-size 8388608 \
	 	--type json

    	# 'elements':
		arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "elements" \
        --on-duplicate "replace" \
		--file "${elements}" \
		--type json

    	# 'reactions':
		arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "reactions" \
        --on-duplicate "replace" \
		--file "${reactions}" \
        --batch-size 8388608 \
		--type json

    	# 'reactionsets':
		arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "reactionsets" \
        --on-duplicate "replace" \
		--file "${reactionsets}" \
		--type json

    	# 'substances':
		arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "substances" \
        --on-duplicate "replace" \
		--file "${substances}" \
        --batch-size 8388608 \
		--type json

        # 'interactions':
        arangoimport \
        --server.endpoint "${serverEndpoint}" \
        --server.username "${userName}" \
        --server.password "${userPassword}" \
        --server.database "${databaseName}" \
        --collection "interactions" \
        --on-duplicate "replace" \
        --file "${interactions}" \
        --batch-size 8388608
        --type json

        # 'mixingmodels':
        arangoimport \
        --server.endpoint "${serverEndpoint}" \
        --server.username "${userName}" \
        --server.password "${userPassword}" \
        --server.database "${databaseName}" \
        --collection "mixingmodels" \
        --on-duplicate "replace" \
        --file "${mixingmodels}" \
        --batch-size 8388608
        --type json

        # 'phases':
        arangoimport \
        --server.endpoint "${serverEndpoint}" \
        --server.username "${userName}" \
        --server.password "${userPassword}" \
        --server.database "${databaseName}" \
        --collection "phases" \
        --on-duplicate "replace" \
        --file "${phases}" \
        --batch-size 8388608
        --type json


    	# 'thermodatasets':
		arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "thermodatasets" \
        --on-duplicate "replace" \
		--file "${thermodatasets}" \
		--type json
	

	# Edge collections 

        # 'basis'
        arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "basis" \
        --on-duplicate "replace" \
		--file "${basis}" \
		--type json

        # 'cites'
        arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "citing" \
        --on-duplicate "replace" \
		--file "${citing}" \
		--type json
    
        # 'defines'
        arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "defines" \
        --on-duplicate "replace" \
		--file "${defines}" \
		--type json
   
        # 'master'
        arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "master" \
        --on-duplicate "replace" \
		--file "${master}" \
		--type json
    
        # 'prodreac'
        arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "prodreac" \
        --on-duplicate "replace" \
		--file "${prodreac}" \
		--type json

        # 'product'
        arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "product" \
        --on-duplicate "replace" \
		--file "${product}" \
		--type json
    
        # 'pulls'
        arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "pulls" \
        --on-duplicate "replace" \
		--file "${pulls}" \
		--type json

        # 'takes'
        arangoimport \
		--server.endpoint "${serverEndpoint}" \
		--server.username "${userName}" \
		--server.password "${userPassword}" \
		--server.database "${databaseName}" \
		--collection "takes" \
        --on-duplicate "replace" \
		--file "${takes}" \
		--type json

        # 'specific'
        arangoimport \
        --server.endpoint "${serverEndpoint}" \
        --server.username "${userName}" \
        --server.password "${userPassword}" \
        --server.database "${databaseName}" \
        --collection "specific" \
        --on-duplicate "replace" \
        --file "${specific}" \
        --type json

        # 'mixing'
        arangoimport \
        --server.endpoint "${serverEndpoint}" \
        --server.username "${userName}" \
        --server.password "${userPassword}" \
        --server.database "${databaseName}" \
        --collection "mixing" \
        --on-duplicate "replace" \
        --file "${mixing}" \
        --type json

        # 'interacts'
        arangoimport \
        --server.endpoint "${serverEndpoint}" \
        --server.username "${userName}" \
        --server.password "${userPassword}" \
        --server.database "${databaseName}" \
        --collection "interacts" \
        --on-duplicate "replace" \
        --file "${interacts}" \
        --type json

        # 'component'
        arangoimport \
        --server.endpoint "${serverEndpoint}" \
        --server.username "${userName}" \
        --server.password "${userPassword}" \
        --server.database "${databaseName}" \
        --collection "component" \
        --on-duplicate "replace" \
        --file "${component}" \
        --type json

#____________________________________[-]_ARANGODB_IMPORT_DATA
