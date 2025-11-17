use std::cmp::Ordering;

fn main() {
    let v = vec![5, 3, 89, 45, 78, 90, 106, 22, 71, 2];
    println!("Before sorting = {:?}", v);
    let v = merge_sort(v);
    // let v = quick_sort(v);
    println!("After sorting = {:?}", v);
}

fn merge_sort(mut v: Vec<i32>) -> Vec<i32> {
    if v.len() <= 1 {
        return v;
    }

    let right = v.split_off(v.len() / 2);
    let mut left = merge_sort(v).into_iter();
    let mut right = merge_sort(right).into_iter();

    let mut merged: Vec<i32> = Vec::with_capacity(left.len() + right.len());

    let mut l_val = left.next();
    let mut r_val = right.next();
    loop {
        if l_val.is_none() && r_val.is_none() {
            break;
        }

        if l_val.is_none() {
            merged.push(r_val.unwrap());
            merged.append(&mut right.collect());
            break;
        }

        if r_val.is_none() {
            merged.push(l_val.unwrap());
            merged.append(&mut left.collect());
            break;
        }

        match l_val.unwrap().cmp(&r_val.unwrap()) {
            Ordering::Greater => {
                merged.push(r_val.unwrap());
                r_val = right.next();
            }
            Ordering::Less => {
                merged.push(l_val.unwrap());
                l_val = left.next();
            }
            Ordering::Equal => (),
        }
    }

    merged
}
