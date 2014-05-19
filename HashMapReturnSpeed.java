import java.util.HashMap;


public class HashMapReturnSpeed {
	
	public static Integer test1(String key, int value){
		HashMap<String,Integer> t1=new HashMap<String,Integer>();
		if(!t1.containsKey(key)){
			t1.put(key, value);
			return t1.get(key);
		}
		return t1.get(key);
	}
	public static Integer test2(String key, int value){
		HashMap<String,Integer> t1=new HashMap<String,Integer>();
		if(!t1.containsKey(key)){
			t1.put(key, value);
			return value;
		}
		return t1.get(key);
	}
	private static int size=100000;
	private static int n=1000;
	public static void main(String[] args){
		long beg1,sum1=0;
		for(int j=0;j<n;++j){
			beg1=System.currentTimeMillis();
			for(int i=0;i<size;++i){
				test1(String.valueOf(i),i);
			}
			sum1+=System.currentTimeMillis()-beg1;
		}
		System.out.println("Old:"+((double)sum1)/n);
		long beg2,sum2=0;
		for(int j=0;j<n;++j){
			beg2=System.currentTimeMillis();
			for(int i=0;i<size;++i){
				test2(String.valueOf(i),i);
			}
			sum2+=System.currentTimeMillis()-beg2;
		}
		System.out.println("New:"+((double)sum2)/n);
		
		/*
		long beg1=System.currentTimeMillis();
		for(int i=0;i<size;++i){
			test1(String.valueOf(i),i);
		}
		System.out.println("Original all fail:"+(System.currentTimeMillis()-beg1));
		beg1=System.currentTimeMillis();
		for(int i=0;i<size;++i){
			test1(String.valueOf(i),i);
		}
		System.out.println("Original all pass:"+(System.currentTimeMillis()-beg1));*/
		
	}
}
