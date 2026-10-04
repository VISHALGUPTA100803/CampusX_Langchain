from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):

    name: str
    age: Optional[int]
    email: EmailStr
    cgpa: float = Field(gt = 0 , lt=10, default=5, description="A decimal value representing the cgpa of the student")

# new_student = {'name': "vishal", 'age':32, 'email':'abc@gmail.com'}

# student = Student (**new_student)

student = Student(
    name="vishal",
    age=32,
    email = "abc@dc.com"
)
print(student)
student_dict = dict(student)
print(student_dict)
print(student_dict["cgpa"])

print(student.model_dump_json())

# ** unpacks a dictionary into keyword arguments.
# Student(**new_student)

# Suppose your dictionary is:
# new_student = {
#     "name": "vishal",
#     "age": 32,
#     "email": "abc@gmail.com"
# }
# When you write:
# student = Student(**new_student)
# Python takes each key as an argument name and its value as the argument value. So the call becomes:
# student = Student(name="vishal", age=32, email="abc@gmail.com")
# These are called keyword arguments because you pass values using names such as name= and age=.
# Without **:
# Student(new_student)
# you’re passing the entire dictionary as one positional argument. Pydantic’s usual model constructor expects named fields, so this call raises an error.