/*
Author: Cao Thanh Tung, Ashwin Nanjappa
Date:   05-Aug-2014

===============================================================================

Copyright (c) 2011, School of Computing, National University of Singapore. 
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/gdel3d.html

If you use gDel3D and you like it or have comments on its usefulness etc., we 
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the National University of Singapore nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#include "DelaunayChecker.h"

#include "gDel3D/CPU/CPUDecl.h"

DelaunayChecker::DelaunayChecker
(
Point3HVec* pointVec,
GDelOutput* output
)
: _pointVec( pointVec ), _output( output )
{
    _predWrapper.init( *pointVec, output->ptInfty );
}

void getTetraSegments( const Tet& t, Segment* sArr )
{
    for ( int i = 0; i < TetSegNum; ++i )
    {
        Segment seg = { t._v[ TetSeg[i][0] ], t._v[ TetSeg[i][1] ] };

        if ( seg._v[0] > seg._v[1] ) std::swap( seg._v[0], seg._v[1] );

        sArr[i] = seg;
    }

    return;
}

void getTetraTriangles( const Tet& t, Tri* triArr )
{
    for ( int i = 0; i < TetFaceNum; ++i )
    {
        // Tri vertices
        Tri tri = { t._v[ TetFace[i][0] ], t._v[ TetFace[i][1] ], t._v[ TetFace[i][2] ] };

        // Sort
        if ( tri._v[0] > tri._v[1] ) std::swap( tri._v[0], tri._v[1] );
        if ( tri._v[1] > tri._v[2] ) std::swap( tri._v[1], tri._v[2] );
        if ( tri._v[0] > tri._v[1] ) std::swap( tri._v[0], tri._v[1] );

        // Add triangle
        triArr[ i ] = tri;
    }

    return;
}

int DelaunayChecker::getVertexCount()
{
    const TetHVec& tetVec       = _output->tetVec;
    const CharHVec& tetInfoVec  = _output->tetInfoVec;

    std::set< int > vertSet;

    // Add vertices
    for ( int ti = 0; ti < tetVec.size(); ++ti )
    {
        if ( !isTetAlive( tetInfoVec[ti] ) ) continue;

        const Tet& tet = tetVec[ ti ];

        vertSet.insert( tet._v, tet._v + 4 );
    }

    return vertSet.size();
}

int DelaunayChecker::getSegmentCount()
{
    const TetHVec& tetVec       = _output->tetVec;
    const CharHVec& tetInfoVec  = _output->tetInfoVec;

    std::set< Segment > segSet;

    // Read segments
    Segment segArr[ TetSegNum ];

    for ( int ti = 0; ti < tetVec.size(); ++ti )
    {
        if ( !isTetAlive( tetInfoVec[ti] ) ) continue;

        const Tet& tet = tetVec[ ti ];

        getTetraSegments( tet, segArr );

        segSet.insert( segArr, segArr + TetSegNum );
    }

    return segSet.size();
}

int DelaunayChecker::getTriangleCount()
{
    const TetHVec& tetVec       = _output->tetVec;
    const CharHVec& tetInfoVec  = _output->tetInfoVec;

    std::set< Tri > triSet;

    // Read triangles
    Tri triArr[ TetFaceNum ];

    for ( int ti = 0; ti < tetVec.size(); ++ti )
    {
        if ( !isTetAlive( tetInfoVec[ti] ) ) continue;

        const Tet& tet = tetVec[ ti ];

        getTetraTriangles( tet, triArr );

        triSet.insert( triArr, triArr + TetFaceNum );
    }

    return triSet.size();
}

int DelaunayChecker::getTetraCount()
{
    const CharHVec& tetInfoVec  = _output->tetInfoVec;

    int count = 0;

    for ( int ti = 0; ti < ( int ) tetInfoVec.size(); ++ti )
        if ( isTetAlive( tetInfoVec[ti] ) )
            ++count;

    return count;
}


void DelaunayChecker::checkEuler()
{
    const int v = getVertexCount();
    std::cout << "V: " << v;

    const int e = getSegmentCount();
    std::cout << " E: " << e;

    const int f = getTriangleCount();
    std::cout << " F: " << f;

    const int t = getTetraCount();
    std::cout << " T: " << t;

    const int euler = v - e + f - t;
    std::cout << " Euler: " << euler << std::endl;

    std::cout << "Euler check: " << ( ( 0 != euler ) ? " ***Fail***" : " Pass" ) << std::endl;

    return;
}

void DelaunayChecker::printTetraAndOpp( int ti, const Tet& tet, const TetOpp& opp )
{
    printf( "TetIdx: %d [ %d %d %d %d ] ( %d:%d %d:%d %d:%d %d:%d )\n",
        ti,
        tet._v[0], tet._v[1], tet._v[2], tet._v[3],
        opp.getOppTet(0), opp.getOppVi(0),
        opp.getOppTet(1), opp.getOppVi(1),
        opp.getOppTet(2), opp.getOppVi(2),
        opp.getOppTet(3), opp.getOppVi(3) );
}

void DelaunayChecker::checkAdjacency()
{
    const TetHVec tetVec        = _output->tetVec;
    const TetOppHVec oppVec     = _output->tetOppVec;
    const CharHVec tetInfoVec   = _output->tetInfoVec;

    for ( int ti0 = 0; ti0 < ( int ) tetVec.size(); ++ti0 )
    {
        if ( !isTetAlive( tetInfoVec[ti0] ) ) continue;

        const Tet& tet0    = tetVec[ ti0 ];
        const TetOpp& opp0 = oppVec[ ti0 ];

        for ( int vi = 0; vi < 4; ++vi )
        {
            if ( -1 == opp0._t[ vi ] ) continue;

            const int ti1   = opp0.getOppTet( vi );
            const int vi0_1 = opp0.getOppVi( vi );

            if ( !isTetAlive( tetInfoVec[ ti1 ] ) )
            {
                std::cout << "TetIdx: " << ti1 << " is invalid!" << std::endl;
                exit(-1);
            }

            const Tet& tet1    = tetVec[ ti1 ];
            const TetOpp& opp1 = oppVec[ ti1 ];

            if ( -1 == opp1._t[ vi0_1 ] || ti0 != opp1.getOppTet( vi0_1 ) )
            {
                std::cout << "Not opp of each other! Tet0: " << ti0 << " Tet1: " << ti1 << std::endl;
                printTetraAndOpp( ti0, tet0, opp0 );
                printTetraAndOpp( ti1, tet1, opp1 );
                exit(-1);
            }

            if ( vi != opp1.getOppVi( vi0_1 ) )
            {
                std::cout << "Vi mismatch! Tet0: " << ti0 << "Tet1: " << ti1 << std::endl;
                exit(-1);
            }
        }
    }

    std::cout << "Adjacency check: Pass\n";

    return;
}

void DelaunayChecker::checkOrientation()
{
    const TetHVec tetVec        = _output->tetVec;
    const CharHVec tetInfoVec   = _output->tetInfoVec;

    int count = 0;

    for ( int i = 0; i < ( int ) tetInfoVec.size(); ++i )
    {
        if ( !isTetAlive( tetInfoVec[i] ) ) continue;

        const Tet& t     = tetVec[i];
        const Orient ord = _predWrapper.doOrient3DAdapt( t._v[0], t._v[1], t._v[2], t._v[3] );

        if ( OrientNeg == ord )
            ++count;
    }

    std::cout << "Orient check: ";
    if ( count )
        std::cout << "***Fail*** Wrong orient: " << count;
    else
        std::cout << "Pass";
    std::cout << "\n";

    return;
}

bool DelaunayChecker::checkDelaunay( bool writeFile )
{
    const TetHVec tetVec        = _output->tetVec;
    const TetOppHVec oppVec     = _output->tetOppVec;
    const CharHVec tetInfoVec   = _output->tetInfoVec;

    const int tetNum = ( int ) tetVec.size();
    int facetSum     = 0;
    int extFacetSum  = 0;

    std::deque< int > que;
    int cont = 0;

    int resp[] = {
        2	,5	,14	,6,
        4	,8	,7	,14,
        10	,2	,1	,12,
        6	,8	,7	,0,
        2	,8	,14	,5,
        6	,0	,11	,12,
        1	,6	,11	,12,
        3	,8	,2	,12,
        2	,6	,1	,12,
        5	,8	,14	,7,
        14	,2	,1	,10,
        2	,8	,6	,12,
        2	,14	,1	,6,
        2	,7	,5	,6,
        4	,8	,13	,0,
        8	,3	,0	,12,
        6	,8	,0	,12,
        0	,3	,11	,12,
        7	,4	,13	,0,
        2	,8	,7	,6,
        2	,8	,5	,7,
        7	,8	,4	,0,
        13	,8	,9	,0,
        0	,3	,8	,9,
        11	,3	,0	,9,
        9	,3	,8	,13
    };

    int resp_count = sizeof(resp) / sizeof(int);
    int resp_tet_count = resp_count / 4;
    std::ofstream oFile2("Failed4.ply");

    auto check_tets = [&](int vv[4]) -> bool{
        for(int n = 0; n < resp_tet_count; n++){
            int addr = 4 * n;
            int matches = 0;
            for(int ti = 0; ti < 4; ti++){
                int ji = resp[addr + ti];
                for(int k = 0; k < 4; k++){
                    if(vv[k] == ji){
                        matches += 1;
                        break;
                    }
                }
            }
            if(matches == 4) return true;
        }
        return false;
    };

    int amount_of_tets = 0;
    for(int i = 0; i < tetVec.size(); i++){
        bool r = true;
        Tet tet = tetVec[i];
        const TetOpp botOpp = oppVec[i];

        if ( !isTetAlive( tetInfoVec[ i ] ) ){
            r = false;
        }

        for(int s = 0; s < 4; s++){
            if(-1 == botOpp._t[s]){
                r = false;
            }
            if(tet._v[s] == 15) r = false;
        }

        if(r){
            amount_of_tets++;
        }
    }

    if(resp_tet_count != amount_of_tets){
        printf("Different number of tets\n");
    }

    oFile2 << "ply\n";
    oFile2 << "format ascii 1.0\n";
    oFile2 << "element vertex " << ( int ) _predWrapper.pointNum() << "\n";
    oFile2 << "property double x\n";
    oFile2 << "property double y\n";
    oFile2 << "property double z\n";
    oFile2 << "element face " << amount_of_tets << "\n";
    oFile2 << "property list uchar int vertex_index\n";
    oFile2 << "end_header\n";
    for ( int i = 0; i < ( int ) _predWrapper.pointNum(); ++i ){
        const Point3 pt = _predWrapper.getPoint( i );

        for ( int vi = 0; vi < 3; ++vi )
            oFile2 << pt._p[ vi ] << " ";
        oFile2 << "\n";
    }

    for(int i = 0; i < tetVec.size(); i++){
        bool r = true;
        Tet tet = tetVec[i];
        const TetOpp botOpp = oppVec[i];

        if ( !isTetAlive( tetInfoVec[ i ] ) ){
            r = false;
        }

        for(int s = 0; s < 4; s++){
            if(-1 == botOpp._t[s]){
                r = false;
            }
            if(tet._v[s] == 15) r = false;
        }

        if(r && resp_tet_count == amount_of_tets){
            bool rr = check_tets(tet._v);
            printf("[%d] = %d %d %d %d\n", (int)rr, tet._v[0],
                    tet._v[1], tet._v[2], tet._v[3]);
            oFile2 << "4 " << tet._v[0] << " " << tet._v[1] << " " <<
                    tet._v[2] << " " << tet._v[3] << " " << std::endl;
            if(!rr){
                printf(" *** INVALID TET\n");
            }
        }
    }

    oFile2.close();

    for ( int botTi = 0; botTi < tetNum; ++botTi )
    {
        bool add = false;
        if ( !isTetAlive( tetInfoVec[ botTi ] ) ) continue;
        cont++;
        const Tet botTet    = tetVec[ botTi ];
        const TetOpp botOpp = oppVec[ botTi ];

        if ( botTet.has( _predWrapper._infIdx ) ){
            add = true;
            extFacetSum++;
        }

        for ( int botVi = 0; botVi < 4; ++botVi ) // Face neighbours
        {
            // No face neighbour
            if ( -1 == botOpp._t[botVi] )
            {

                ++facetSum;
                continue;
            }

            const int topVi = botOpp.getOppVi( botVi );
            const int topTi = botOpp.getOppTet( botVi );

            if ( topTi < botTi ) continue; // Neighbour will check
            if(true){
                int eentry = ( botTi << 2 ) | botVi;
                que.push_back( eentry );
            }

            ++facetSum;

            const Tet topTet  = tetVec[ topTi ];
            const int topVert = topTet._v[ topVi ];

            Side side = _predWrapper.doInsphereAdapt( botTet, topVert );

            if ( SideIn != side ) continue;

            int entry = ( botTi << 2 ) | botVi;
            que.push_back( entry );

            if ( !botOpp.isOppSphereFail( botVi ) &&
                !oppVec[ topTi ].isOppSphereFail( topVi ) )
            {
                std::cout << "********** Fail: " << botTi << " " << botVi << " "
                    << topTi << " " << topVi << std::endl;

                const TetOpp opp = oppVec[ topTi ];
                const int *ordVi = TetViAsSeenFrom[ topVi ];

                for ( int i = 0; i < 3; ++i )
                {
                    const int sideTi = opp.getOppTet( ordVi[i] );

                    if ( botOpp.isNeighbor( sideTi ) )
                        std::cout << "3-2 flip: " << sideTi << std::endl;
                }
            }
        }
    }

    std::cout << "\nConvex hull facets: " << extFacetSum << std::endl;
    std::cout << "\nDelaunay check: ";

    if ( que.empty() )
    {
        std::cout << "Pass" << std::endl;
        return true;
    }

    std::cout << "***Fail*** Failed faces: " << que.size() << std::endl;

    if ( writeFile )
    {
        // Write failed facets to file

        std::cout << "Writing failures to file ... ";

        const int pointNum = ( int ) _predWrapper.pointNum();
        const int triNum   = ( int ) que.size();

        std::ofstream oFile( "Failed.ply" );

        oFile << "ply\n";
        oFile << "format ascii 1.0\n";
        oFile << "element vertex " << pointNum << "\n";
        oFile << "property double x\n";
        oFile << "property double y\n";
        oFile << "property double z\n";
        oFile << "element face " << triNum << "\n";
        oFile << "property list uchar int vertex_index\n";
        oFile << "end_header\n";

        // Write points

        for ( int i = 0; i < pointNum; ++i )
        {
            const Point3 pt = _predWrapper.getPoint( i );
            printf("[%d] = [%g %g %g]\n", i, pt._p[0], pt._p[1], pt._p[2]);

            for ( int vi = 0; vi < 3; ++vi )
                oFile << pt._p[ vi ] << " ";
            oFile << "\n";
        }

        // Write failed faces

        for ( int fi = 0; fi < triNum; ++fi )
        {
            const int entry = que[ fi ];
            const int tvi   = entry & 3;
            const int ti    = entry >> 2;

            const Tet tet      = tetVec[ ti ];
            const int* orderVi = TetViAsSeenFrom[ tvi ];

            bool inf = false;
            for(int faceI = 0; faceI < 3; ++faceI){
                inf |= tet._v[orderVi[ faceI ]] == pointNum-1;
            }

            if(inf){
                continue;
            }
            oFile << "3 ";
            for ( int faceI = 0; faceI < 3; ++faceI )
                oFile << tet._v[ orderVi[ faceI ] ] << " ";
            oFile << "\n";
        }

        std::cout << " done!\n";
    }

    return false;
}

