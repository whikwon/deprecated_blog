---
title:  "Python_class"
date: 2017-08-21 00:00:00
comments: true
---

Python의 class(클래스)에 대한 내용을 학습하기 위해ㅇㅇ [블로그](https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/)
를 요약 및 번역하였다.

```python
class Customer(object):
    """A customer of ABC Bank with a checking account. Customers have the
    following properties:

    Attributes:
        name: A string representing the customer's name.
        balance: A float tracking the current balance of the customer's account.
    """

    def __init__(self, name, balance=0.0):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.name = name
        self.balance = balance

    def withdraw(self, amount):
        """Return the balance remaining after withdrawing *amount*
        dollars."""
        if amount > self.balance:
            raise RuntimeError('Amount greater than available balance.')
        self.balance -= amount
        return self.balance

    def deposit(self, amount):
        """Return the balance remaining after depositing *amount*
        dollars."""
        self.balance += amount
        return self.balance
```

### 1) class(클래스) <br>
위 코드에서 Customer라는 클래스를 만들긴 했지만 실제 사람 고객 정보를 만든 것은 아니고 어떤 청사진을
만들어 놓은 것이다. 클래스를 실행시켰을 때 (**e.g. jeff = Customer('Jeff', 1000)**)
입력되는 jeff라는 object가 실제 고객 정보라고 생각하면 되겠다. Customer는
클래스는 실제 고객 정보(object)는 아니며 수많은 인스턴스들을 만들어낼 수 있는 어떤 틀이라고 생각하면 쉽겠다.

### 2) self <br>
self는 Customer 클래스의 instance이다. 모든 매서드에 다 포함되어 있고 우리가 위에서 선언한
jeff라는 instance가 클래스 내 함수를 실행할 수 있도록 해준다. (**e.g. jeff.withdraw(100)**)
그리고 위에 보면 매서드 내 인스턴스에 self가 붙어있는데, 이로 하여금 클래스 내에서 변수가 공유되도록 해준다.
self가 없을 경우 해당 인스턴스는 포함된 매서드를 벗어나지 못한다.

### 3) __init__ <br>
initializer라고 생각하면 되겠다. 클래스를 실행시켰을 때 입력해주어야 할 모든 정보들을
__init__ 함수에서 선언해주면 된다. 그리고 __init__ 에서 따로 변수를 지정하고 또 다른 매서드에서
변수를 따로 지정하지 않고 최대한 __init__ 에서 해결하는게 좋다.

### 4) Instance Attributes and Methods <br>
위의 예시에서 클래스 내부에서 실행되는 매서드나 공유되는 인스턴스가 self를 필요로 했던 것을 보았다.
이런 인스턴스와 매서드를 **instance attributes, method** 라고 부른다. 이와 대비되는 ***Static Method***
를 아래에서 추가로 살펴보도록 하자.

  - **Static Methods** <br>
  인스턴스 매서드와 다른 **Class attributes, method** 를 소개하도록 하겠다.
  아래 예시를 보면 **wheels** 라는 인스턴스는 self가 없이도 mustang에서 해당 값에 접근할 수 있다.
  이렇듯 attribute가 *class-level* 에 있다고 해서 *Class attributes* 라고 한다.   
  ```python
  class Car(object):

    wheels = 4

    def __init__(self, make, model):
        self.make = make
        self.model = model

    mustang = Car('Ford', 'Mustang')
    print mustang.wheels
    # 4
    print Car.wheels
    # 4
    ```
    그럼 ***Class method*** 도 살펴보도록 하자.
    아래 코드를 보면 *make_car_sound* 매서드에 **self** 가 없다. 이 클래스를 인스턴스에 넣고
    매서드를 실행시키면 에러가 발생하는 것을 확인할 수 있다.
    그럼 실행이 되도록 하려면 어떻게 해야 될까?
    ```python
    class Car(object):
        ...
        def make_car_sound():
            print('VRooooommmm!')
    ```
    위에서 **self** 없이도 인스턴스에 접근이 가능했던 것처럼 매서드도 가능하게 만들어 줄 수 있다.
    방법은 **@staticmethod** 를 사용하는 것이다.
    ```python
    class Car(object):
        ...
        @staticmethod
        def make_car_sound():
            print('VRooooommmm!')
    ```

### 5) Inheritance(상속) <br>
상속은 Object-Oriented Programming에서 매우 유용하고 중요한 역할을 한다.
간단하게는 문자 그대로 **parent** 클래스의 행동을 **child** 클래스가 물려받는 것이다.
상세한 내용은 아래 예시를 보면서 더 설명하도록 하겠다.

```python
class Car(object):
    """A car for sale by Jeffco Car Dealership.

    Attributes:
        wheels: An integer representing the number of wheels the car has.
        miles: The integral number of miles driven on the car.
        make: The make of the car as a string.
        model: The model of the car as a string.
        year: The integral year the car was built.
        sold_on: The date the vehicle was sold.
    """

    def __init__(self, wheels, miles, make, model, year, sold_on):
        """Return a new Car object."""
        self.wheels = wheels
        self.miles = miles
        self.make = make
        self.model = model
        self.year = year
        self.sold_on = sold_on

    def sale_price(self):
        """Return the sale price for this car as a float amount."""
        if self.sold_on is not None:
            return 0.0  # Already sold
        return 5000.0 * self.wheels

    def purchase_price(self):
        """Return the price for which we would pay to purchase the car."""
        if self.sold_on is None:
            return 0.0  # Not yet sold
        return 8000 - (.10 * self.miles)
```

```python
class Truck(object):
    """A truck for sale by Jeffco Car Dealership.

    Attributes:
        wheels: An integer representing the number of wheels the truck has.
        miles: The integral number of miles driven on the truck.
        make: The make of the truck as a string.
        model: The model of the truck as a string.
        year: The integral year the truck was built.
        sold_on: The date the vehicle was sold.
    """

    def __init__(self, wheels, miles, make, model, year, sold_on):
        """Return a new Truck object."""
        self.wheels = wheels
        self.miles = miles
        self.make = make
        self.model = model
        self.year = year
        self.sold_on = sold_on

    def sale_price(self):
        """Return the sale price for this truck as a float amount."""
        if self.sold_on is not None:
            return 0.0  # Already sold
        return 5000.0 * self.wheels

    def purchase_price(self):
        """Return the price for which we would pay to purchase the truck."""
        if self.sold_on is None:
            return 0.0  # Not yet sold
        return 10000 - (.10 * self.miles)

```
위 2개 코드를 보면 처음에 Car를 팔기 위해서 Car라는 클래스를 만들었고 Truck을 더 팔기 위해서
Truck이라는 클래스를 만들었다. 두 개의 코드를 비교해봤을 때 내용이 거의 다 같다는 점을
알 수 있다. 음..여기에서 오토바이가 추가되고 자전거가 추가되고 이러면 **중복된 코드** 의
양이 매우 많아져서 비효율적이게 될 것이다.
(*programming에서 중요한 원칙 중에 하나는 반복하지 말라는 것이다.*)

그럼 공통된 부분을 합칠 방법에 대해서 생각해보도록 하자. 그게 바로 **상속** 의 개념이다.
공통된 부분을 합친 아래 Vehicle 클래스를 보자. 살펴보면 **완전히 같은 부분에 대해서**
Vehicle이라는 Car, Truck의 개념을 포함하는 클래스를 만들어서 포함시켰다. 그리고 이렇게 만든
클래스를 Car, Truck 클래스를 선언할 때 포함시켜주면(*e.g. Car(Vehicle)*) 상속 받아서 잘 작동한다.

그리고 purchase_price 매서드의 경우 다른 부분이 있었는데 어떻게 처리했는 지 아래 Car, Truck
클래스와 함께보면서 살펴보도록 하자.

```python
from abc import ABCMeta, abstractmethod
class Vehicle(object):
    """A vehicle for sale by Jeffco Car Dealership.


    Attributes:
        wheels: An integer representing the number of wheels the vehicle has.
        miles: The integral number of miles driven on the vehicle.
        make: The make of the vehicle as a string.
        model: The model of the vehicle as a string.
        year: The integral year the vehicle was built.
        sold_on: The date the vehicle was sold.
    """

    __metaclass__ = ABCMeta

    base_sale_price = 0
    wheels = 0

    def __init__(self, miles, make, model, year, sold_on):
        self.miles = miles
        self.make = make
        self.model = model
        self.year = year
        self.sold_on = sold_on

    def sale_price(self):
        """Return the sale price for this vehicle as a float amount."""
        if self.sold_on is not None:
            return 0.0  # Already sold
        return 5000.0 * self.wheels

    def purchase_price(self):
        """Return the price for which we would pay to purchase the vehicle."""
        if self.sold_on is None:
            return 0.0  # Not yet sold
        return self.base_sale_price - (.10 * self.miles)

    @abstractmethod
    def vehicle_type(self):
        """"Return a string representing the type of vehicle this is."""
        pass    
```    
**Car** 와 **Truck** 의 클래스는 아래와 같다.
**purchase_price** 매서드의 경우 클래스 별 가격이 달라 base_sale_price라는 새로운
인스턴스를 만든 후에 Car, Truck 각 클래스에 Class attribute로 선언해주었다.
그리고 추가로 **@abstractmethod(추상클래스)** 이 사용되었다. 추상클래스는
Car, Truck이 유사한 점이 많아 공통된 Vehicle로부터 상속을 받을 때, 코드가 복잡해지거나
각각의 type이 헷갈릴 경우가 발생할 수도 있어서 이를 매서드로 정의해주고 넘어가게 만드는 것이다.
아래 코드에서 Car 클래스를 선언할 때 **vehicle_type** 매서드를 선언하지 않으면 에러가 발생한다.  
이렇듯 각기 다른 클래스에서 특정한 매서드를 **꼭** 실행시켜주고 싶을 때 추상클래스를 사용하며
좀 더 상세한 추상클래스의 활용법에 대해서는 나중에 알아보기로 하자.

```python
class Car(Vehicle):
    """A car for sale by Jeffco Car Dealership."""

    base_sale_price = 8000
    wheels = 4

    def vehicle_type(self):
        """"Return a string representing the type of vehicle this is."""
        return 'car'

class Truck(Vehicle):
    """A truck for sale by Jeffco Car Dealership."""

    base_sale_price = 10000
    wheels = 4

    def vehicle_type(self):
        """"Return a string representing the type of vehicle this is."""
        return 'truck'
```
여기서 **Motorcycle** 클래스를 더 추가하고 싶다면 아래 코드를 작성하면 된다. 매우 쉽고 효율적이지 않나.
```python
class Motorcycle(Vehicle):
    """A motorcycle for sale by Jeffco Car Dealership."""

    base_sale_price = 4000
    wheels = 2

    def vehicle_type(self):
        """"Return a string representing the type of vehicle this is."""
        return 'motorcycle'
```

```python
class Date(object):

    def __init__(self, day=0, month=0, year=0):
        self.day = day
        self.month = month
        self.year = year

    @classmethod
    def from_string(cls, date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        date1 = cls(day, month, year)
        return date1

    @staticmethod
    def is_date_valid(date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        return day <= 31 and month <= 12 and year <= 3999
```

추가적으로 stackoverflow에서 좋은 예제를 찾아서 **classmethod, staticmethod** 사용에
대한 부연 설명을 하고자 한다. (*위 코드 참조*)<br>
**classmethod** 는 위 코드처럼 형식이 다른 (e.g. 'dd-mm-yyyy') 데이터가 입력되었을 때에도
Date class를 만들고 싶은 경우에 사용한다. 형식이 하나면 좋겠지만 여러가지가 있을 것이고 일일이
데이터를 가공해주는 일보다는 classmethod를 사용해서 하는 것이 좀 더 OOP에 맞다.
(*@classmethod 의 매서드는 재사용 가능하고 cls가 class를 받기 때문에 상속도 가능하다.*)

**staticmethod** 는 classmethod와 유사하면서도 좀 다르다. (*self가 필요없지*)
주로 위에서 처럼 *instantiation(인스턴스화)가 필요하지 않은 경우에 사용된다.*
이게 무슨 말이냐면 위에서 입력된 date문자가 유효한 값인지 확인하는데 사용되었는데
이 때, 어떤 인스턴스도 해당 매서드에 입력되지 않았다는게 중요하다. classmethod나 instance method를
사용할 경우에 인스턴스를 계속해서 입력해주어야 하므로 staticmethod에 비해 효율적이지 못하게 된다.


Reference:  <br>
https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/
https://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner
