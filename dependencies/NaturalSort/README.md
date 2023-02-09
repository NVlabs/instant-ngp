# NaturalSort ![CI](https://github.com/scopeInfinity/NaturalSort/workflows/CI/badge.svg)
C++ Header File for Natural Comparison and Natural Sort


##### Calling Methods 

*  __For Natural Sorting__

		void SI::natural::sort(Container<String>);
		
		void SI::natural::sort(IteratorBegin<String>,IteratorEnd<String>);
		
		void SI::natural::sort<String,CArraySize>(CArray<String>);
		
	
*  __For Natural Comparision__

		bool SI::natural::compare<String>(String lhs,String rhs);
		bool SI::natural::compare<String>(char *const lhs,char *const rhs);
	
	Here we can have
			
			std::vector<std::string> 	as Container<String>
			String 					as std::string
			CArray<String>			as std::string[CArraySize]
		
	
	
		

#####  Example

*  __Inputs__

		Hello 100
		Hello 34
		Hello 9
		Hello 25
		Hello 10
		Hello 8

*  __Normal Sort Output__

		Hello 10
		Hello 100
		Hello 25
		Hello 34
		Hello 8
		Hello 9

*  __Natural Sort Output__

		Hello 8
		Hello 9
		Hello 10
		Hello 25
		Hello 34
		Hello 100



