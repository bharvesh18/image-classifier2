function firstNonConsecutive (arr) {
    for(let i=0;i<arr.length;i++){
        if(arr[i+1]===arr[i]+1){
            continue
        }
        else{
            console.log(arr[i+1]);
            break;
        }
    }
}
firstNonConsecutive([1,2,3,4,6,7,8])